from abc import abstractmethod
import warnings

import numpy as np
import torch

from mxmc.util.generic_numerical_optimization \
    import perform_slsqp_then_nelder_mead
from .acv_constraints import satisfies_constraints
from mxmc.optimizers.optimizer_base import OptimizerBase
from mxmc.optimizers.optimizer_base import OptimizationResult
from mxmc.sample_allocations.acv_sample_allocation import ACVSampleAllocation

TORCHDTYPE = torch.double


class ACVOptimizer(OptimizerBase):

    def __init__(self, model_costs, covariance=None, recursion_refs=None):
        super().__init__(model_costs, covariance)
        self._covariance_tensor = self._create_covariance_tensor()

        self._model_costs_tensor = torch.tensor(self._model_costs,
                                                dtype=TORCHDTYPE)
        if recursion_refs is None:
            recursion_refs = [0] * (self._num_models - 1)
        self._recursion_refs = recursion_refs

        self._alloc_class = ACVSampleAllocation

    def _create_covariance_tensor(self):
        covariance = torch.tensor(self._covariance, dtype=TORCHDTYPE)
        ndim = covariance.ndimension()
        if ndim == 2:
            return covariance.unsqueeze(0)
        if ndim == 3:
            return covariance.permute([2, 0, 1])
        raise RuntimeError("Invalid Covariance matrix encountered with "
                           "dimension =", str(ndim))

    def optimize(self, target_cost, variance_function=None):
        '''
        Variance Function
        Input: Torch Tensor, size (:,Noutput,Noutput)
        Output: Scalar
        NEEDS TO BE COMPATABLE WITH TORCH AUTOGRAD
        '''
        
        if variance_function is None:
            variance_function = lambda x: x.diagonal(dim1=-1,dim2=-2).sum(-1)
        
        if target_cost < np.sum(self._model_costs):
            return self._get_invalid_result()
        if self._num_models == 1:
            return self._get_monte_carlo_result(target_cost,variance_function)

        ratios = self._solve_opt_problem(target_cost, variance_function)

        sample_nums = self._compute_sample_nums_from_ratios(ratios,
                                                            target_cost)
        sample_nums = np.floor(sample_nums)

        variance = self._compute_variance_from_sample_nums(sample_nums, variance_function)
        actual_cost = self._compute_total_cost_from_sample_nums(sample_nums)

        comp_allocation = self._make_allocation(sample_nums)

        allocation = self._alloc_class(comp_allocation)

        return OptimizationResult(actual_cost, variance, allocation)

    def _solve_opt_problem(self, target_cost, variance_function):
        bounds = self._get_bounds()
        constraints = self._get_constraints(target_cost)
        initial_guess = self._get_initial_guess(constraints)

        def obj_func(rat):
            return self._compute_objective_function(rat, target_cost, variance_function,
                                                    gradient=False)

        def obj_func_and_grad(rat):
            return self._compute_objective_function(rat, target_cost, variance_function,
                                                    gradient=True)

        ratios = perform_slsqp_then_nelder_mead(bounds, constraints,
                                                initial_guess, obj_func,
                                                obj_func_and_grad)

        return ratios

    def _get_initial_guess(self, constraints):
        # balanced_costs = self._model_costs[0] / self._model_costs[1:]
        # if satisfies_constraints(balanced_costs, constraints):
        #     return balanced_costs
        #
        # all_ones = np.ones(self._num_models - 1)
        # if satisfies_constraints(all_ones, constraints):
        #     return all_ones
        #
        # all_twos = np.full(self._num_models - 1, 2)
        # if satisfies_constraints(all_twos, constraints):
        #     return all_twos

        increasing_values = np.arange(2, self._num_models + 1)
        if not satisfies_constraints(increasing_values, constraints):
            warnings.warn("Could not identify an initial guess that satisfies"
                          " constraints")
        return increasing_values

        # full_initial_guess = np.ones(self._num_models)
        # for _ in range(self._num_models):
        #     full_initial_guess[1:] = \
        #         full_initial_guess[self._recursion_refs] + 1
        # recursion_initial_guess = full_initial_guess[1:]
        # if satisfies_constraints(recursion_initial_guess, constraints):
        #     return recursion_initial_guess
        #
        # full_initial_guess = np.ones(self._num_models)
        # for _ in range(self._num_models):
        #     full_initial_guess[1:] = \
        #         full_initial_guess[self._recursion_refs] + 1
        #     full_initial_guess[1:] = \
        #         full_initial_guess[self._recursion_refs] - 1
        # full_initial_guess[1:] = \
        #     full_initial_guess[self._recursion_refs] + 1
        # recursion_initial_guess = full_initial_guess[1:]
        # if satisfies_constraints(recursion_initial_guess, constraints):
        #     return recursion_initial_guess

        # raise RuntimeError("Could not identify an initial guess that
        # satisfies constraints")

    def _compute_objective_function(self, ratios, target_cost, variance_function, gradient):
        ratios_tensor = torch.tensor(ratios, requires_grad=gradient,
                                     dtype=TORCHDTYPE)
        N = self._calculate_n_autodiff(ratios_tensor, target_cost)

        try:
            variance = \
                self._compute_acv_estimator_variance(self._covariance_tensor,
                                                     ratios_tensor, N, variance_function)
            variance = variance.sum()
        except RuntimeError:
            variance = 9e99 * torch.dot(ratios_tensor, ratios_tensor)

        if not gradient:
            return variance.detach().numpy()

        variance.backward()
        result = (variance.detach().numpy(),
                  ratios_tensor.grad.detach().numpy())
        return result

    def _calculate_n(self, ratios, target_cost):
        eval_ratios = self._get_model_eval_ratios(ratios)
        N = target_cost / (np.dot(self._model_costs, eval_ratios))
        return N

    def _calculate_n_autodiff(self, ratios_tensor, target_cost):
        eval_ratios = self._get_model_eval_ratios_autodiff(ratios_tensor)
        N = target_cost / (torch.dot(self._model_costs_tensor, eval_ratios))
        return N

    def _compute_acv_estimator_variance(self, covariance, ratios, N, variance_function):
        # Tbig_C = covariance[:, self._num_outputs:, self._num_outputs:]
        # Tc_bar = covariance[:, 0, self._num_outputs:] \
        #     / torch.sqrt(covariance[:, 0, 0]).unsqueeze(1)

        # TF, TF0 = self._compute_acv_F_and_F0(ratios)
        # Ta = (TF0.unsqueeze(0) * Tc_bar)

        # Talpha = torch.linalg.solve(Tbig_C * TF, Ta.unsqueeze(2))
        # TR_squared = (Ta.unsqueeze(2) * Talpha).sum(1).sum(1)
        # Tvariance = covariance[:, 0, 0] / N * (1 - TR_squared)
        # return Tvariance

        big_C = covariance[:, self._num_outputs:, self._num_outputs:]
        c = covariance[:, :self._num_outputs, self._num_outputs:]

        F, F0 = self._compute_acv_F_and_F0(ratios)
        F0 = torch.repeat_interleave(F0,self._num_outputs,dim=0)
        F0 = torch.repeat_interleave(F0.unsqueeze(0),self._num_outputs,dim=0)
        F = torch.repeat_interleave(F,self._num_outputs,dim=0)
        F = torch.repeat_interleave(F,self._num_outputs,dim=1)
        
        H = (F0.unsqueeze(0) * c)

        # alpha = torch.matmul(H, torch.linalg.inv(big_C * F.unsqueeze(0)))
        alpha = torch.transpose(torch.linalg.solve(torch.transpose(big_C * F.unsqueeze(0),1,2), torch.transpose(H,1,2)),1,2)
        inv_high_var = torch.linalg.inv(covariance[:,:self._num_outputs,:self._num_outputs])
        
        pre_R_squared = torch.matmul(alpha,torch.transpose(H,1,2))
        R_squared = torch.matmul(inv_high_var, pre_R_squared)
        variance = torch.matmul(covariance[:,:self._num_outputs,:self._num_outputs] / N, (torch.eye(self._num_outputs).unsqueeze(0) - R_squared))

        #This will need to be a weighted trace in the future
        # return variance.diagonal(dim1=-1,dim2=-2).sum(-1)
        out = variance_function(variance)
        return out

    def _compute_sample_nums_from_ratios(self, ratios, target_cost):
        N = self._calculate_n(ratios, target_cost)
        sample_nums = N * np.array([1] + list(ratios))
        return sample_nums

    def _compute_variance_from_sample_nums(self, sample_nums, variance_function):
        N = sample_nums[0]
        ratios = sample_nums[1:] / N
        ratios_tensor = torch.tensor(ratios, dtype=TORCHDTYPE)
        variance = self._compute_acv_estimator_variance(
                self._covariance_tensor, ratios_tensor, N, variance_function)
        variance = variance.detach().numpy()
        if len(variance) == 1:
            return variance[0]
        return variance

    def _compute_total_cost_from_sample_nums(self, sample_nums):
        N = sample_nums[0]
        ratios = sample_nums[1:] / N
        eval_samples_nums = N * self._get_model_eval_ratios(ratios)
        cost = np.dot(eval_samples_nums, self._model_costs)
        return cost

    @abstractmethod
    def _get_bounds(self):
        raise NotImplementedError

    @abstractmethod
    def _get_constraints(self, target_cost):
        raise NotImplementedError

    @abstractmethod
    def _compute_acv_F_and_F0(self, ratios):
        raise NotImplementedError

    @abstractmethod
    def _make_allocation(self, sample_nums):
        raise NotImplementedError

    @abstractmethod
    def _get_model_eval_ratios(self, ratios):
        raise NotImplementedError

    @abstractmethod
    def _get_model_eval_ratios_autodiff(self, ratios_tensor):
        raise NotImplementedError
