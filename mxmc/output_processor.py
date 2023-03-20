import numpy as np


class OutputProcessor:
    '''
    Class to estimate covariance matrix from collection of output samples from
    multiple models. Handles the general case where each models' outputs were
    not computed from the same set of random inputs and pairwise overlaps
    between samples are identified using a SampleAllocation object.
    '''
    def __init__(self):
        pass

    @staticmethod
    def compute_covariance_matrix(model_outputs, sample_allocation=None):
        '''
        Estimate covariance matrix from collection of output samples from
        multiple models. In the simple case, the outputs for each model were
        generated from the same collection of inputs and covariance can be
        straightforwardly computed. In the general case, the outputs were not
        computed from the same inputs and a SampleAllocation object must be
        supplied to identify overlap between samples to compute covariance.

        :param model_outputs: list of arrays of outputs for each model. Each
            array must be the same size unless a sample allocation is provided.
        :type model_outputs: list of np.array
        :param sample_allocation: An MXMC sample allocation object defining the
            indices of samples that each model output was generated for, if
            applicable. Default is None indicating that all supplied model
            output arrays are the same size and were generated for the same
            inputs.
        :type sample_allocation: SampleAllocation object.

        :Returns: covariance matrix among all model outputs (2D np.array with
            size equal to the number of models).
        '''
        if len(model_outputs[0].shape) == 2:
            nvariables = model_outputs[0].shape[0]
        else:
            nvariables = 1

        output_array = OutputProcessor._build_output_array(model_outputs,
                                                           sample_allocation, nvariables)
        cov_matrix = OutputProcessor._compute_cov_elements(output_array)
        return np.array(cov_matrix)

    @staticmethod
    def _build_output_array(model_outputs, sample_allocation, nvariables):
        if sample_allocation is None:
            if nvariables > 1:
                model_inds = [list(range(len(out[0]))) for out in model_outputs]
            else:
                model_inds = [list(range(len(out))) for out in model_outputs]
        else:
            model_inds = [sample_allocation.get_sample_indices_for_model(i)
                          for i in range(len(model_outputs))]

        return OutputProcessor._make_output_array_from_indices(
            model_inds, model_outputs,nvariables)

    @staticmethod
    def _make_output_array_from_indices(model_inds, model_outputs, nvariables):
        '''
        At one index of model_outputs, 
        Each row should be a variable, and each column
        a single observation of those variables, similarly to np.cov()
        '''
            
        max_model_inds = [max(i) for i in model_inds if i]

        if not max_model_inds:
            return np.empty((0, 0))

        max_ind = max(max_model_inds)
        if nvariables > 1:
            indeces_list = []
            variable_list = np.full((nvariables*len(model_outputs), max_ind + 1), np.nan)
            for ii in range(len(model_outputs)):
                for jj in range(nvariables):
                    variable_list[ii*nvariables+jj] = model_outputs[ii][jj]
                    indeces_list.append(model_inds[ii])
        else:
            variable_list = model_outputs
            indeces_list = model_inds
            
        output_array = np.full((nvariables*len(model_outputs), max_ind + 1), np.nan)
        for i, (inds, out) in enumerate(zip(indeces_list, variable_list)):
            output_array[i, inds] = out
        return output_array

    @staticmethod
    def _compute_cov_elements(output_array):
        num_totvars = output_array.shape[0]
        matrix = np.full((num_totvars, num_totvars), np.nan)
        for i in range(num_totvars):
            for j in range(i, num_totvars):
                filter_ = np.logical_and(~np.isnan(output_array[i]),
                                         ~np.isnan(output_array[j]))
                if len(output_array[i, filter_]) <= 1:
                    continue
                element_matrix = np.cov(output_array[i, filter_],
                                        output_array[j, filter_],
                                        ddof=1)[0, 1]
                matrix[i, j] = element_matrix
                matrix[j, i] = element_matrix

        return matrix
