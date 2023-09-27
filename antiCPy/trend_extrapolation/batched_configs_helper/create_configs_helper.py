import numpy as np
from scipy.special import binom

def construct_start_combinations_helper(data, total_data, tuple_num, pick_out_combination):
    """
    Internal helper method to construct the first CP configuration of a certain batch. It assumes to draw the
    ``pick_out_combination`` from a total list of configurations in a systematic combinatoric order.
    """
    maximum_digit_range = total_data - tuple_num + 1
    data_step = data[1] - data[0]
    ordered_segment_sizes = np.zeros(maximum_digit_range)
    sub_pick_out_combination = np.copy(pick_out_combination)
    config_building_blocks = np.copy(data[:maximum_digit_range])
    combination = np.array([])
    already_drawn = 0
    for i in range(tuple_num):
        num_binomial_coeffs = maximum_digit_range - (already_drawn - i)
        ordered_segment_sizes = np.zeros(num_binomial_coeffs)
        config_building_blocks = np.copy(config_building_blocks[:num_binomial_coeffs])
        for k in range(num_binomial_coeffs):
            ordered_segment_sizes[k] = binom(total_data - (already_drawn + k + 1), tuple_num - i - 1)
        ordered_segment_sizes = np.cumsum(ordered_segment_sizes)
        lower_orientation_bound = np.append([0], ordered_segment_sizes[:-1])
        segment_bool = np.less_equal.outer(sub_pick_out_combination, ordered_segment_sizes) & np.greater.outer(
            sub_pick_out_combination, lower_orientation_bound)

        low_bound_index = 0
        for y in range(segment_bool.size):
            if segment_bool[y]:
                low_bound_index = y
                break
        sub_pick_out_combination = sub_pick_out_combination - lower_orientation_bound[low_bound_index]
        combination = np.append(combination, config_building_blocks[segment_bool])
        for y in range(data.size):
            if data[y] == combination[-1]:
                already_drawn = y + 1
                break
        if low_bound_index > 0:
            building_block_update = (low_bound_index + 1) * data_step
        elif low_bound_index == 0:
            building_block_update = data_step
        config_building_blocks = config_building_blocks + building_block_update
    return combination


def extrapolate_batch_combinations(data, batch_size, tuple_num, pick_out_combination):
    """
    Internal helper method to extrapolate the next ``batch_size`` CP configurations starting with the one constructed by
    the helper method ``construct_start_combinations_helper`` in systematic order.
    """
    total_data = data.size
    start_combination = construct_start_combinations_helper(data, total_data, tuple_num, pick_out_combination)
    batch_configurations = np.zeros((batch_size, tuple_num))
    dx = data[1] - data[0]
    batch_configurations[0, :] = start_combination
    maximum_data = np.copy(data[total_data - tuple_num:])
    for i in range(batch_size - 1):
        if batch_configurations[i, -1] < maximum_data[-1]:
            counter_array = np.zeros(tuple_num)
            counter_array[-1] = dx
            batch_configurations[i + 1, :] = batch_configurations[i, :] + counter_array
        else:
            reset_size = 1
            for y in range(1, tuple_num):
                if batch_configurations[i, -y] == maximum_data[-y]:
                    reset_size += 1
            batch_configurations[i + 1, :-(reset_size)] = batch_configurations[i, :-(reset_size)]
            for y in range(reset_size):
                batch_configurations[i + 1, tuple_num - reset_size + y] = batch_configurations[
                                                                              i, tuple_num - reset_size] + (
                                                                                  y + 1) * dx
    return batch_configurations


def batched_configs(batch_num, batchsize, x, prediction_horizon, n_cp, exact_sum_control=False,
                    config_output=False):
    """
    Internal helper method to initialize the CP configuration of a given batch.
    """
    if exact_sum_control:
        possible_configs = extrapolate_batch_combinations(x[1:-1], batchsize, n_cp, batch_num * batchsize + 1)
        if config_output:
            print('Possible configs: ', possible_configs)
            print('Possible configs shape: ', possible_configs.shape)
        start_value_dummy = np.ones((possible_configs.shape[0], 1)) * x[0]
        if prediction_horizon == None:
            end_value_dummy = np.ones((possible_configs.shape[0], 1)) * x[-1]
        elif prediction_horizon > 0:
            end_value_dummy = np.ones((possible_configs.shape[0], 1)) * prediction_horizon
        composition_dummy = np.append(start_value_dummy, np.append(possible_configs, end_value_dummy, axis=1),
                                      axis=1)
        if config_output:
            print('MC_cp_configurations: ', composition_dummy)
            print('MC_cp_configurations shape: ', composition_dummy.shape)
        MC_cp_configurations = composition_dummy
    elif exact_sum_control == False:
        support_x = np.zeros((batchsize, n_cp + 2))
        support_x[:, 1:] = - np.log(1 - np.random.uniform(low=0.0001, high=1.0, size=(batchsize, n_cp + 1)))
        support_y = np.zeros((batchsize, n_cp + 2))
        z_cp_config = np.zeros((batchsize, n_cp + 2))
        for k in range(n_cp + 2):
            support_y[:, k] = support_x[:, k] / (np.sum(support_x, axis=1))
        for k in range(n_cp + 2):
            z_cp_config[:, k] = np.sum(support_y[:, :k + 1], axis=1)
        z_cp_config[:, 0] = x[0]
        if prediction_horizon == None:
            z_cp_config[:, -1] = x[-1]
        elif prediction_horizon > 0:
            z_cp_config[:, -1] = prediction_horizon
        z_cp_config[:, 1:-1] = x[0] + z_cp_config[:, 1:-1] * (x[-1] - x[0])
        MC_cp_configurations = z_cp_config
    return MC_cp_configurations
