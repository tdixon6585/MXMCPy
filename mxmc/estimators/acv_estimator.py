import numpy as np

from .estimator_base import EstimatorBase


class ACVEstimator(EstimatorBase):
    """
    Class to create ACV estimators given an optimal sample allocation and
    outputs from high & low fidelity models.

    :param allocation: ACVSampleAllocation object defining the optimal sample
            allocation using an ACV optimizer.
    :type allocation: ACVSampleAllocation object
    :param covariance: Covariance matrix defining covariance among all
            models being used for estimator. Size MxM where M is # models.
    :type covariance: 2D np.array
    """
    def __init__(self, allocation, covariance):
        super().__init__(allocation, covariance)
        self._cov_delta_delta, self._cov_q_delta = \
            self._calculate_cov_delta_terms()
        self._alpha = self._calculate_alpha()

    def get_estimate(self, model_outputs):
        self._validate_model_outputs(model_outputs)
        q = np.mean(model_outputs[0],axis=1)
        for i in range(1, self._allocation.num_models):
            ranges_1, ranges_2 = self._allocation.get_sample_split_for_model(i)
            n_1 = sum([len(i) for i in ranges_1])
            n_2 = sum([len(i) for i in ranges_2])
            for rng in ranges_1:
                q += np.dot(self._alpha[:,(i-1)*self._num_outputs:i*self._num_outputs], np.sum(model_outputs[i][:,rng],axis=1)) / n_1
            for rng in ranges_2:
                q -= np.dot(self._alpha[:,(i-1)*self._num_outputs:i*self._num_outputs], np.sum(model_outputs[i][:,rng],axis=1)) / n_2
        
        return q

    def _calculate_cov_delta_terms(self):
        k_0 = self._allocation.get_k0_matrix()
        k = self._allocation.get_k_matrix()
        
        k_0 = np.repeat(np.repeat([k_0],self._num_outputs, axis=0),self._num_outputs,axis=1)
        k = np.repeat(np.repeat(k,self._num_outputs, axis=0),self._num_outputs,axis=1)
        
        cov_q_delta = k_0 * self._covariance[:self._num_outputs, self._num_outputs:]
        cov_delta_delta = k * self._covariance[self._num_outputs:, self._num_outputs:]
        return cov_delta_delta, cov_q_delta

    def _get_approximate_variance(self):
        n_0 = self._allocation.get_number_of_samples_per_model()[0]
        var_q0 = self._covariance[0, 0]

        variance = var_q0 / n_0 \
            + self._alpha.dot(self._cov_delta_delta.dot(self._alpha)) \
            + 2 * self._alpha.dot(self._cov_q_delta)

        return variance

    def _calculate_alpha(self):
        k_indices = [i - 1 for i in self._allocation.utilized_models if i != 0]
        all_indices = np.arange(0,self._num_outputs*self._allocation.num_models).reshape((self._allocation.num_models,self._num_outputs))
        cov_indices = all_indices[k_indices].flatten()
        # if matrix is None:
        #     return None
        # return np.copy(matrix[cov_indices][:, cov_indices])
        
        temp_cov_delta_delta = self._cov_delta_delta[cov_indices][:, cov_indices]
        temp_cov_q_delta = self._cov_q_delta[:,cov_indices]
        alpha = np.zeros((self._num_outputs, (self._allocation.num_models - 1)*self._num_outputs))
        alpha[:,cov_indices] = - np.linalg.solve(temp_cov_delta_delta,
                                             temp_cov_q_delta.T).T
        return alpha
