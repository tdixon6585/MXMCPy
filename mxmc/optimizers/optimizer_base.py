from abc import ABCMeta, abstractmethod
from collections import namedtuple

import torch

import numpy as np


OptimizationResult = namedtuple('OptResult',
                                ['cost', 'variance', 'allocation'])


class InconsistentModelError(Exception):
    pass


class OptimizerBase(metaclass=ABCMeta):

    def __init__(self, model_costs, covariance=None, *_, **__):
        self._model_costs = np.array(model_costs)
        self._num_models = len(self._model_costs)
        self._covariance = covariance
        self._num_outputs = int(len(covariance)/self._num_models)

        if covariance is not None:
            self._validate_covariance_matrix(covariance)

        self._alloc_class = None

    def _validate_covariance_matrix(self, matrix):
        if len(matrix) != self._num_outputs*self._num_models:
            error_msg = "Covariance matrix and model cost dims must match"
            raise ValueError(error_msg)

        matrix_t = matrix.transpose([1, 0] + list(range(2, matrix.ndim)))
        if not np.allclose(matrix_t, matrix):
            error_msg = "Covariance matrix array must be symmetric"
            raise ValueError(error_msg)

    @abstractmethod
    def optimize(self, target_cost, *_, **__):
        raise NotImplementedError

    def subset(self, model_indices):
        subset_costs = np.copy(self._model_costs[model_indices])
        subset_covariance = self._get_subset_of_matrix(self._covariance,
                                                       model_indices,self._num_outputs,self._num_models)
        return self.__class__(subset_costs, subset_covariance)

    @staticmethod
    def _get_subset_of_matrix(matrix, model_indices,num_outputs,num_models):
        all_indices = np.arange(0,num_outputs*num_models).reshape((num_models,num_outputs))
        cov_indices = all_indices[model_indices].flatten()
        if matrix is None:
            return None
        return np.copy(matrix[cov_indices][:, cov_indices])

    def get_num_models(self):
        return self._num_models

    def _get_invalid_result(self):
        allocation = np.zeros((1, 2 * self._num_models), dtype=int)
        allocation[0, :2] = 1
        return OptimizationResult(0, np.inf, self._alloc_class(allocation))

    def _get_monte_carlo_result(self, target_cost,variance_function):
        sample_nums = np.floor(np.array([target_cost / self._model_costs[0]]))
        # variance = self._covariance[0, 0] / sample_nums[0]
        variance = variance_function(torch.Tensor(self._covariance[:self._num_outputs,:self._num_outputs]).unsqueeze(0)) / sample_nums[0]
        cost = self._model_costs * sample_nums[0]
        allocation = np.zeros((1, 2 * self._num_models), dtype=int)
        allocation[0, 0] = sample_nums[0]
        allocation[0, 1] = 1
        return OptimizationResult(cost, variance,
                                  self._alloc_class(allocation))
