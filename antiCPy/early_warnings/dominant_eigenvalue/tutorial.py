from antiCPy.early_warnings import dominant_eigenvalue
import numpy as np
import os

def example(threshold_series = False):

	# load the data
	this_dir, this_filename = os.path.split(__file__)
	DATA_PATH1 = os.path.join(this_dir, "data", "ricker_type.csv")
	gendata_ricker_type=np.genfromtxt(DATA_PATH1, delimiter=',')
	# create time sampling
	time_ricker_type = np.arange(gendata_ricker_type.size)
	# optimize embedding dimension with a time consuming, but detailed threshold series.
	if threshold_series == True :
		fnn_ricker_type = dominant_eigenvalue.param_opt.various_R_threshold_fnn(gendata_ricker_type, start_order = 1, end_order = 15, start_threshold = 15, end_threshold = 50)
		dominant_eigenvalue.graphics.plot_fnn(fnn_ricker_type)
	# otimize embedding dimension with a fast one threshold analysis
	fnn_ricker_type_II = dominant_eigenvalue.param_opt.false_NN(gendata_ricker_type)
	dominant_eigenvalue.graphics.plot_fnn(fnn_ricker_type_II, R_threshold_series = False, R_threshold = '30', start_order = 1, end_order = 15)
	# otimize time delay
	tau_distances = dominant_eigenvalue.param_opt.avg_distance_from_diagonal(gendata_ricker_type, E = 3, start_lag = 1, end_lag = 10, image = False)
	dominant_eigenvalue.graphics.plot_avg_DD(tau_distances)
	# estimate the absolute dominant eigenvalues and the eigenvalues per window
	A,B = dominant_eigenvalue.analysis.AR_EV_calc(gendata_ricker_type, 1200, 3)
	# plot the absolute dominant eigenvalue trend with the investigated dataset
	dominant_eigenvalue.graphics.abs_max_eigval_plot(A, time_ricker_type, gendata_ricker_type, cl_1 = [9000, 'b'], ws_1 = 1200, axis = [0,10000,0.75,1.1], integrated_plot = True)
	# plot the dominant eigenvalues in the complex plane.
	dominant_eigenvalue.graphics.max_eigval_gauss_plot(B, label_1 = 'Ricker-type model')

def load_data():
	this_dir, this_filename = os.path.split(__file__)
	DATA_PATH1 = os.path.join(this_dir, "data", "ricker_type.csv")
	DATA_PATH2 = os.path.join(this_dir, "data", "henon_normal.csv")
	DATA_PATH3 = os.path.join(this_dir, "data", "hopf_normal.csv")
	a=np.genfromtxt(DATA_PATH1, delimiter=',')
	b=np.genfromtxt(DATA_PATH2, delimiter =',')
	c=np.genfromtxt(DATA_PATH3, delimiter =',')
	return a, b, c