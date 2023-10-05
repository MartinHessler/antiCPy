# -*- coding: utf-8 -*-

"""
The ``eigval_tsa.param_opt`` module contains basic functions to optimize the parameters that are needed for an eigenvalue estimation with the module ``eigval_tsa.analysis``. The functions

	#. ``embedding_attractor_reconstruction(data, E, index_shift)``
	#. ``false_NN(data, time, index_shift, start_order = 1, end_order = 15, NN_threshold = 30)``
	#. ``various_R_threshold_fnn(data, index_shift= 1, start_threshold = 15, end_threshold = 50, start_order = 1, end_order = 15, save = False, save_name = 'fnn_R_threshold_series_default00.npy')``
	#. ``avg_distance_from_diagonal(data, E, start_lag = 1, end_lag = 10, image = False)``

are explained in the following.
"""

import numpy as np
import matplotlib.pylab as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean

def embedding_attractor_reconstruction(data, E, index_shift = 1):
	"""
	:param data: The time series data that should be embedded.

	:param E: The dimension of the desired delayed state space embedding of the investigated time series.

	:param index_shift: This parameter represents the desired time delay for the time delayed embedding. An index shift of one stands for one sampling step of the given time series, an index shift of two stands for a delay of two sampling steps of the time series per embedding dimension. The default value is one.

	:type data: The time series is given as a one-dimensional numpy array.

	:type E: The dimension :math:`E` should be an integer :math:`E \in \mathbb{N} \setminus \{ 0, 1\}`.

	:type index_shift: The index shift is given as an integer :math:`index` _ :math:`shift` :math:`\in \mathbb{N}\setminus \{ 0 \}` .

	:return: The result is a two dimensional numpy array that represents the time delayed state space reconstruction of the time series with the entered dimension :math:`E` and the time delay :math:`index` _ :math:`shift`. Every column represents a vector state in state space.

	:rtype: The result is given as a two dimensional numpy array.
	"""

	time_lagged_matrix=np.zeros((E, data.size-(E-1)*index_shift))

	for i in range(E):
		time_lagged_matrix[i,:]=np.roll(data, shift=i*index_shift)[(data.size-(data.size-(E-1)*index_shift)):]

	return time_lagged_matrix



def various_R_threshold_fnn(data, index_shift= 1, start_threshold = 15, end_threshold = 50, start_order = 1, end_order = 15, save = False, save_name = 'fnn_R_threshold_series_default00.npy'):
	"""
	:param data: The time series data of which an optimal time delayed embedding dimension is needed.
	:param index_shift: This parameter represents the desired time delay for the time delayed embedding. An index shift of one stands for one sampling step of the given time series, an index shift of two stands for a delay of two sampling steps of the time series per embedding dimension. The default value is one.
	:param start_threshold: The threshold value determines the distance in the dimension :math:`m+1` between a pair of next neighours in the dimension :math:`m`. The threshold value should be chosen big. A reasonable start threshold value is 15.
	:param end_threshold: The threshold value determines the distance in the dimension :math:`m+1` between a pair of next neighours in the dimension :math:`m`. The threshold value should be chosen big. A reasonable end threshold value is 50.
	:param start_order: The start embedding dimension for which  the number of false next neighbours is calculated. The default value is one.
	:param end_order: The last embedding dimension for which  the number of false next neighbours is calculated. The default value is 15.
	:param save: Determines wether the fnn data is saved as a numpy array or not. Default is ``False``.
	:param save_name: Specifies the name of the saved fnn data array.
	:type data: One dimensional numpy array.
	:type index_shift: Integer.
	:type start_threshold: Integer.
	:type end_threshold: Integer.
	:type start_order: Integer.
	:type end_order: Integer.
	:type save: Boolean.
	:type save_name: String.
	:return: The result is a two dimensional numpy array with the counted number of false next neighbours for the chosen threshold values. The rows correspond to subsequent thresholds. The first row corresponds to ``start_threshold``. The last row corresponds to ``end_threshold``. The columns correspond to subsequent embedding dimensions. The first entry contains the number of false next neighbours of the ``start_dimension``, the last entry is given by the number of false next neighbours of the ``end_dimension``.
	:rtype: The result is given as a two dimensional numpy array.
	:note: The NN_threshold interval is chosen following the guide line in [KBA92]_ .
	.. [KBA92] Matthew B. Kennel, Reggie Brown, and Henry D. I. Abarbanel. “Determining embedding dimension for phase-space reconstruction using a geometrical construction”. In: Phys. Rev. A 45 (6 Mar. 1992), pp. 3403–3411. doi: 10.1103/ PhysRevA . 45 . 3403. url: https : / / link . aps . org / doi / 10 . 1103 / PhysRevA.45.3403.
	"""
	
	false_NN_plot_data = np.zeros((end_threshold - start_threshold +1 , end_order-start_order +1)) # allocate space for the fnn R threshold series

	for i in range(start_threshold, end_threshold + 1):
		false_NN_plot_data[i - start_threshold,:]=false_NN(data, index_shift= 1, start_order = start_order, end_order = end_order, NN_threshold = i)

	if save == True:
		np.save(save_name, false_NN_plot_data)
	return false_NN_plot_data



def false_NN(data, index_shift=1, start_order = 1, end_order = 15, NN_threshold = 30):
	"""
	:param data: The time series data of which an optimal time delayed embedding dimension is needed.
	:param index_shift: This parameter represents the desired time delay for the time delayed embedding. An index shift of one stands for one sampling step of the given time series, an index shift of two stands for a delay of two sampling steps of the time series per embedding dimension. The default value is one.
	:param start_order: The start embedding dimension for which  the number of false next neighbours is calculated. The default value is one.
	:param end_order: The last embedding dimension for which  the number of false next neighbours is calculated. The default value is 15.
	:param NN_threshold: The threshold value determines the distance in the dimension :math:`m+1` between a pair of next neighours in the dimension :math:`m`. The threshold value should be chosen big. A reasonable range of threshold values lies in the interval [15, 50].
	:type data: The time series is given as a one-dimensional numpy array.
	:type index_shift: This parameter represents the desired time delay for the time delayed embedding. An index shift of one stands for one sampling step of the given time series, an index shift of two stands for a delay of two sampling steps of the time series per embedding dimension. The default value is one.
	:type start_order: This parameter is an integer :math:`start` _ :math:`order` :math:`\in \mathbb{N}\setminus \{ 0 \}` . The default value is one.
	:type end_order: This parameter is an integer :math:`end` _ :math:`order` :math:`\in \mathbb{N}\setminus \{ 0 \}` . It should hold :math:`end` _ :math:`order>start` _ :math:`order` . The default value is 15.
	:type NN_threshold: The threshold value is an integer that is commonly chosen as :math:`NN` _ :math:`threshold \in [15, 50 ]` .
	:return: The result is a one dimensional numpy array with the counted number of false next neighbours for the chosen threshold value. The first entry contains the number of false next neighbours of the start_dimension, the last entry is given by the number of false next neighbours of the end_dimension.
	:rtype: The result is given as a one dimensional numpy array.
	:note: The NN_threshold interval is chosen following the guide line in [KBA92]_ .
	
	"""

	def false_NN(data, time, index_shift, start_order=1, end_order=15, NN_threshold=30):

		sigma_deviation = np.std(data)
		false_NN = np.zeros(end_order + 1 - start_order)
		j = 0
		for m in range(start_order, end_order + 1):
			state_space_m = embedding_attractor_reconstruction(data, time, m, index_shift)
			state_space_m_plus_1 = embedding_attractor_reconstruction(data, time, m + 1, index_shift)
			state_space_m_T = np.transpose(state_space_m)
			state_space_m_plus_1_T = np.transpose(state_space_m_plus_1)
			nbrs_model_m = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(state_space_m_T)
			euclidean_dists_m, NN_indices_m = nbrs_model_m.kneighbors(state_space_m_T)
			euclidean_dists_m_plus_1_of_NN_m = np.zeros(state_space_m_T[:, 0].size - index_shift)
			heaviside_arg_false_NN = np.zeros(euclidean_dists_m_plus_1_of_NN_m.size)
			for i in range(euclidean_dists_m_plus_1_of_NN_m.size):
				euclidean_dists_m_plus_1_of_NN_m[i] = euclidean(
					state_space_m_plus_1_T[NN_indices_m[i + index_shift, 1] - index_shift, :], state_space_m_plus_1_T[i,
																							   :])  # Durch die Erhöhung der Einbettung verschiebt sich der usprüngliche Index der Dimension m um -1. Ansonsten soll für alle i Zustände der false_NN durchgeführt werden.
			for i in range(heaviside_arg_false_NN.size):
				if euclidean_dists_m[i + index_shift, 1] != 0:
					heaviside_arg_false_NN[i] = (euclidean_dists_m_plus_1_of_NN_m[i] / euclidean_dists_m[
						(i + index_shift), 1]) - NN_threshold
				elif euclidean_dists_m_plus_1_of_NN_m[i] != 0:
					heaviside_arg_false_NN[i] = 1
				else:
					heaviside_arg_false_NN[i] = 1  # Following Kantz, Schreiber
			heaviside_arg_filter = (sigma_deviation / NN_threshold) - euclidean_dists_m[index_shift:, 1]
			if np.sum(np.heaviside(heaviside_arg_filter, 0)) != 0:
				false_NN[j] = np.sum(
					np.heaviside(heaviside_arg_false_NN, 0) * np.heaviside(heaviside_arg_filter, 0)) / (
								  np.sum(np.heaviside(heaviside_arg_filter, 0)))
			else:
				false_NN[j] = 0
			j += 1
		return false_NN



def avg_distance_from_diagonal(data, E, start_lag = 1, end_lag = 10, image = False):
	"""
	:param data: The time series data for which is needed an optimal time delayed embedding dimension.
	:param E: The dimension of the desired delayed state space embedding of the investigated time series.
	:param start_lag: The start time lag that is chosen to calculate the average distance from diagonal for a given dimension :math:`E` . The default value is one.
	:param end_lag: The last time lag for which is calculated the average distance from diagonal for a given dimension :math:`E` . The default value is 10.
	:param image: A boolean with the default value False. If it is chosen True, the function creates a plot of the average distance from diagonal depending on the chosen time lag.
	:type data: The time series is given as a one-dimensional numpy array.
	:type E: The dimension :math:`E` should be an integer :math:`E \in \mathbb{N} \setminus \{ 0, 1\}` .
	:type start_lag: This parameter is an integer :math:`start` _ :math:`lag` :math:`\in \mathbb{N}\setminus \{ 0 \}` . The default value is one.
	:type end_lag: This parameter is an integer :math:`end` _ :math:`lag` :math:`\in \mathbb{N}\setminus \{ 0 \}` . It should hold :math:`end` _ :math:`lag>start` _ :math:`lag` . The default value is 10.
	:type image: Boolean variable.
	:return: The function returns a one dimensional numpy array with the average distances from diagonal for every integer time lag in the chosen interval. The average distance from diagonal for the start time lag can be found in the first array entry. The array is designed in ascending time lags.
	:rtype: The result is given as a one dimensional numpy array.
	"""
	dummy_time=np.zeros(2)
	S_tau = np.zeros(end_lag - start_lag + 1)
	for i in range(start_lag, end_lag + 1):
		lagged_data = embedding_attractor_reconstruction(data, E, i)
		M = lagged_data[0,:].size
		distances_squared = np.zeros((E, M))
		for j in range(1,E):
			distances_squared[j,:] = (lagged_data[j,:] - lagged_data[0,:])**2
		S_tau[i-1] = 1./ M * np.sum(np.sqrt(np.sum(distances_squared, axis =0)))
		#print(S_tau)

	if image == True:
		x_axis_tau = np.arange(start_lag, end_lag + 1, 1)
		print(x_axis_tau)
		ax = plt.subplot(111)
		ax.plot(x_axis_tau, S_tau, 'bv-', label = 'average distance from diagonal')
		plt.axis([0, end_lag,0, 1 ])
		plt.xlabel('time lag $\tau$')
		plt.ylabel('<S>')
		plt.legend()
		plt.show()
	return S_tau