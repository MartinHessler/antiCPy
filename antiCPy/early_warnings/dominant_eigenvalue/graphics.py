# -*- coding: utf-8 -*-

"""
The ``eigval_tsa.graphics`` module contains basic functions to plot the results of an eigenvalue estimation with the module ``eigval_tsa.param_opt`` and ``eigval_tsa.analysis`` toolkits. The functions

	#. ``abs_max_eigval_plot(...)``
	#. ``prep_plot_imaginary_plane(eigvals)``
	#. ``max_eigval_gauss_plot(...)``

are explained in the following.
"""



import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def abs_max_eigval_plot(ev_1 = [], time_1 = [], data_1 = [], cl_1 = ['','b'], ev_2 = [], time_2 = [], data_2 = [], cl_2 = ['', 'g'], ev_3 = [], time_3 = [], data_3 = [], cl_3 = ['', 'r'], ev_4 = [], time_4 = [], data_4 = [], cl_4 = ['', 'k'], ev_5 = [], time_5 = [], data_5 = [], cl_5 = ['','m'], ev_6 = [], time_6 = [], data_6 = [], cl_6 = ['','c'], ws_1 = 100, ws_2 = 100, ws_3 = 100, ws_4 = 100, ws_5 = 100, ws_6 = 100, label_1 = '', label_2 = '', label_3 = '', label_4 = '', label_5 = '', label_6 = '', ls_1 = 'b-', ls_2 = 'g-', ls_3 = 'r-', ls_4 = 'k-', ls_5 = 'm-', ls_6 = 'c-', axis = [0,100,-1.1,1.1], integrated_plot = False, save = False, save_name = 'default00.png', show = True):
	"""
	:param ev_1: A one dimensional numpy array containing an estimated absolute eigenvalue time series.
	:param time_1: A one dimensional numpy array containing the the time sampling of the investigated time series of which one has estimated the eigenvalues.
	:param data_1: A one dimensional numpy array containing the time series of which one has estimated the eigenvalues.
	:param cl_1: A one dimensional string array with two entries. The first contains the time at which a critical transition shall be marked with a dotted vertical critical line. If it is empty as by default, no line is plotted. The second entry is a string with the color options of the line. 
	:param ev_2: A one dimensional numpy array containing an estimated absolute eigenvalue time series.
	:param time_2: A one dimensional numpy array containing the the time sampling of the investigated time series of which one has estimated the eigenvalues.
	:param data_2: A one dimensional numpy array containing the time series of which one has estimated the eigenvalues.
	:param cl_2: A one dimensional string array with two entries. The first contains the time at which a critical transition shall be marked with a dotted vertical critical line. If it is empty as by default, no line is plotted. The second entry is a string with the color options of the line.
	:param ev_3: A one dimensional numpy array containing an estimated absolute eigenvalue time series.
	:param time_3: A one dimensional numpy array containing the the time sampling of the investigated time series of which one has estimated the eigenvalues.
	:param data_3: A one dimensional numpy array containing the time series of which one has estimated the eigenvalues.
	:param cl_3: A one dimensional string array with two entries. The first contains the time at which a critical transition shall be marked with a dotted vertical critical line. If it is empty as by default, no line is plotted. The second entry is a string with the color options of the line.
	:param ev_4: A one dimensional numpy array containing an estimated absolute eigenvalue time series.
	:param time_4: A one dimensional numpy array containing the the time sampling of the investigated time series of which one has estimated the eigenvalues.
	:param data_4: A one dimensional numpy array containing the time series of which one has estimated the eigenvalues.
	:param cl_4: A one dimensional string array with two entries. The first contains the time at which a critical transition shall be marked with a dotted vertical critical line. If it is empty as by default, no line is plotted. The second entry is a string with the color options of the line.
	:param ev_5: A one dimensional numpy array containing an estimated absolute eigenvalue time series.
	:param time_5: A one dimensional numpy array containing the the time sampling of the investigated time series of which one has estimated the eigenvalues.
	:param data_5: A one dimensional numpy array containing the time series of which one has estimated the eigenvalues.
	:param cl_5: A one dimensional string array with two entries. The first contains the time at which a critical transition shall be marked with a dotted vertical critical line. If it is empty as by default, no line is plotted. The second entry is a string with the color options of the line.
	:param ev_6: A one dimensional numpy array containing an estimated absolute eigenvalue time series.
	:param time_6: A one dimensional numpy array containing the the time sampling of the investigated time series of which one has estimated the eigenvalues.
	:param data_6: A one dimensional numpy array containing the time series of which one has estimated the eigenvalues.
	:param cl_6: A one dimensional string array with two entries. The first contains the time at which a critical transition shall be marked with a dotted vertical critical line. If it is empty as by default, no line is plotted. The second entry is a string with the color options of the line.
	:param ws_1: An integer number that indicates the rolling window size of the first set of ev_1, time_1 and data_1.
	:param ws_2: An integer number that indicates the rolling window size of the second set of ev_2, time_2 and data_2.
	:param ws_3: An integer number that indicates the rolling window size of the third set of ev_3, time_3 and data_3.
	:param ws_4: An integer number that indicates the rolling window size of the fourth set of ev_4, time_4 and data_4.
	:param ws_5: An integer number that indicates the rolling window size of the fifth set of ev_5, time_5 and data_5.
	:param ws_6: An integer number that indicates the rolling window size of the sixth set of ev_6, time_6 and data_6.
	:param ls_1: The line style of dataset 1 is by default 'b-'.
	:param ls_2: The line style of dataset 2 is by default 'g-'.
	:param ls_3: The line style of dataset 3 is by default 'r-'.
	:param ls_4: The line style of dataset 4 is by default 'k-'.
	:param ls_5: The line style of dataset 5 is by default 'm-'.
	:param ls_6: The line style of dataset 6 is by default 'c-'.
	:param label_1: A string with the label for the ev_1.
	:param label_2: A string with the label for the ev_2.
	:param label_3: A string with the label for the ev_3.
	:param label_4: A string with the label for the ev_4.
	:param label_5: A string with the label for the ev_5.
	:param label_6: A string with the label for the ev_6.
	:param axis: A one dimensional array with four limiting entries for the x and y axis of the plot. The first two entries are the lower and upper x limit, the last two entries are the lower and upper y limit. The array is [0,100,-1.1,1.1] by default.
	:param integrated_plot: If the boolean is ``True``, two plots are generated in a window. The upper one shows the eigenvalue time series, the lower one shows the investigated dataset. If the boolean is ``False``, just the upper plot is generated.
	:param save: If the boolean is ``True``, the plot is saved under the name of ``save_name``.
	:param save_name: The string is 'default00.png' by default.
	:param show: If the boolean is ``True``, the plot is shown. Otherwise it is not.
	:type ev_1: One dimensional numpy array of floats.
	:type ev_2: One dimensional numpy array of floats.
	:type ev_3: One dimensional numpy array of floats.
	:type ev_4: One dimensional numpy array of floats.
	:type ev_5: One dimensional numpy array of floats.
	:type ev_6: One dimensional numpy array of floats.
	:type data_1: One dimensional numpy array of floats.
	:type data_2: One dimensional numpy array of floats.
	:type data_3: One dimensional numpy array of floats.
	:type data_4: One dimensional numpy array of floats.
	:type data_5: One dimensional numpy array of floats.
	:type data_6: One dimensional numpy array of floats.
	:type cl_1: One dimensional string array with two entries.
	:type cl_2: One dimensional string array with two entries.
	:type cl_3: One dimensional string array with two entries.
	:type cl_4: One dimensional string array with two entries.
	:type cl_5: One dimensional string array with two entries.
	:type cl_6: One dimensional string array with two entries.
	:type time_1: One dimensional numpy array of floats.
	:type time_2: One dimensional numpy array of floats.
	:type time_3: One dimensional numpy array of floats.
	:type time_4: One dimensional numpy array of floats.
	:type time_5: One dimensional numpy array of floats.
	:type time_6: One dimensional numpy array of floats.
	:type label_1: String.
	:type label_2: String.
	:type label_3: String.
	:type label_4: String.
	:type label_5: String.
	:type label_6: String.
	:type ws_1: Integer.
	:type ws_2: Integer.
	:type ws_3: Integer.
	:type ws_4: Integer.
	:type ws_5: Integer.
	:type ws_6: Integer.
	:type ls_1: String with the specific linestyle encoding of ``matplotlib.plot``.
	:type ls_2: String with the specific linestyle encoding of ``matplotlib.plot``.
	:type ls_3: String with the specific linestyle encoding of ``matplotlib.plot``.
	:type ls_4: String with the specific linestyle encoding of ``matplotlib.plot``.
	:type ls_5: String with the specific linestyle encoding of ``matplotlib.plot``.
	:type ls_6: String with the specific linestyle encoding of ``matplotlib.plot``.
	:type axis: A one dimensional array with four float entries.
	:type integrated_plot: Boolean.
	:type save: Boolean.
	:type save_name: String.
	:type show: Boolean.
	:return: The function generates a plot with the desired features. Depending on the input the function shows and saves the plot.
	"""

	if integrated_plot == False:
		for i in range(1,7):
			if eval("ev_" + str(i)) != []:
				last_window_point = eval("time_" + str(i))[eval("ws_" + str(i)) - 1:]
				plt.figure()
				max_eigvals_plot=plt.subplot(111)
				max_eigvals_plot.plot(last_window_point, eval("ev_" + str(i)), eval("ls_" + str(i)), label = eval("label_" + str(i)))
				if eval('cl_' + str(i))[0] != '':
					max_eigvals_plot.axvline(x= float(eval("cl_" + str(i))[0]), c = eval("cl_" + str(i))[1], ls=':')
			max_eigvals_plot.plot(axis[0:2], np.ones(2), c='orange', ls = '--')
		plt.xlabel(r'time $t$', fontsize = 18)
		plt.ylabel(r'$|\lambda_{\rm max} |$', fontsize = 18)
		plt.axis(axis)
		plt.legend(fontsize = 15)
		plt.tight_layout()
		if save == True:
			plt.savefig(save_name)
		if show == True:
			plt.show()

	if integrated_plot == True:
		for i in range(1,7):
			if eval("ev_" + str(i)) != []:
				last_window_point = eval("time_" + str(i))[eval("ws_" + str(i)) - 1:]
				plt.figure()
				max_eigvals_plot=plt.subplot(211)
				plt.axis(axis)
				plt.ylabel(r'$|\lambda_{\rm max} |$', fontsize = 18)
				data_plot=plt.subplot(212)
				plt.xlim(axis[0],axis[1])
				max_eigvals_plot.plot(last_window_point, eval("ev_" + str(i)), eval("ls_" + str(i)), label = eval("label_" + str(i)))
				data_plot.plot(eval("time_" + str(i)), eval("data_" + str(i)), eval("ls_" + str(i)), label = eval("label_" + str(i)))
				if eval('cl_' + str(i))[0] != '':
					max_eigvals_plot.axvline(x= float(eval("cl_" + str(i))[0]), c = eval("cl_" + str(i))[1], ls=':')
			max_eigvals_plot.plot(axis[0:2], np.ones(2), c='orange', ls = '--')
		plt.xlabel('time [a.u.]',  fontsize = 18)
		plt.ylabel('data [a.u.]', fontsize = 18)
		plt.legend(fontsize = 15)
		plt.tight_layout()
		if save == True:
			plt.savefig(save_name)
		if show == True:
			plt.show()


def prep_plot_imaginary_plane(eigvals):
	"""
	:param eigvals: A two dimensional numpy array that contains all estimated eigenvalues for a specific rolling window in a row. Each following row corresponds to the eigenvalues in the next rolling window.
	:type eigvals: Two dimensional numpy array of complex floats.
	:return: The function returns a one dimensional numpy array of the eigenvalues of each rolling window that have maximum absolute value in each rolling time window. In the case of complex conjugated eigenvalues both eigenvalues are appended to the output.
	:rtype: One dimensional numpy array of complex float numbers.
	"""
	max_eigvals=eigvals[0,np.where(np.abs(eigvals[0,:])==np.max(np.abs(eigvals[0,:])))]
	for i in range(1,eigvals[:,0].size):
		max_eigvals=np.append(max_eigvals, eigvals[i,np.where(np.abs(eigvals[i,:])==np.max(np.abs(eigvals[i,:])))])
	return max_eigvals


def max_eigval_gauss_plot(ev_1 =[], ev_2 =[], ev_3 =[], label_1 ='', label_2 = '', label_3 = '', cmap_1 = 'Blues', cmap_2 = 'Greens', cmap_3 = 'Reds', title = '', save = False, save_name = 'gaussian_default00.png', show = True):
	"""
	:param ev_1: A one dimensional numpy array containing the maximum estimated eigenvalues of each rolling time window that are eventually obtained and sorted by ``eigval_tsa.graphics.prep_plot_imaginary_plane``.
	:param ev_2: A one dimensional numpy array containing the maximum estimated eigenvalues of each rolling time window that are eventually obtained and sorted by ``eigval_tsa.graphics.prep_plot_imaginary_plane``.
	:param ev_3: A one dimensional numpy array containing the maximum estimated eigenvalues of each rolling time window that are eventually obtained and sorted by ``eigval_tsa.graphics.prep_plot_imaginary_plane``.
	:param label_1: String with the legend name of ev_1.
	:param label_2: String with the legend name of ev_2.
	:param label_3: String with the legend name of ev_3.
	:param cmap_1: The colour map to illustrate the time evolution in the Gaussian plane for the ev_1 is by default the string 'Blues'.
	:param cmap_2: The colour map to illustrate the time evolution in the Gaussian plane for the ev_2 is by default the string 'Greens'.
	:param cmap_3: The colour map to illustrate the time evolution in the Gaussian plane for the ev_3 is by default the string 'Reds'.
	:param title: String with the plot's title. Default is empty.
	:param save: The Boolean is ``False`` by default. If it is ``True``, the plot is saved with the name of the string save_name.
	:param save_name: String name of the saved plot. It is 'gaussian_default00.png' by default.
	:param show: The Boolean is ``True`` by default and the generated plot is shown. Otherwise it will not be shown.
	:type ev_1: One dimensional numpy array of complex floats.
	:type ev_2: One dimensional numpy array of complex floats.
	:type ev_3: One dimensional numpy array of complex floats.
	:type label_1: String.
	:type label_2: String.
	:type label_3: String.
	:type cmap_1: String with the specific encoding of colour maps in ``matplotlib.scatter`` with ``Colormap`` from ``matplotlib.colors``.
	:type cmap_2: String with the specific encoding of colour maps in ``matplotlib.scatter`` with ``Colormap`` from ``matplotlib.colors``.
	:type cmap_3: String with the specific encoding of colour maps in ``matplotlib.scatter`` with ``Colormap`` from ``matplotlib.colors``.
	:type save: Boolean.
	:type save_name: String.
	:type show: Boolean.
	:return: The function generates a time resolved plot of the eigenvalues in the Gaussian plane which have  maximum absolute value. It can be shown and saved as required.

	"""
	x=np.linspace(0,2*np.pi,10000)
	plt.figure()
	imag_plot=plt.subplot(111)

	for i in range(1,4):
		if eval("ev_" + str(i)) != []:
			prepared_ev=prep_plot_imaginary_plane(eval("ev_" + str(i)))
			imag_plot.scatter(np.real(prepared_ev),np.imag(prepared_ev),c = np.arange(prepared_ev.size), cmap = eval("cmap_" + str(i)), edgecolors = 'none',label= eval("label_" + str(i)))

	imag_plot.plot(np.sin(x),np.cos(x),'k-')
	imag_plot.axis('equal')
	plt.tight_layout()
	plt.xlabel(r'$\Re (\hat{\lambda}_{\rm max})$', fontsize = 18)
	plt.ylabel(r'$\Im (\hat{\lambda}_{\rm max})$',  fontsize = 18)
	plt.title(title)
	plt.legend(fontsize = 15)
	plt.tight_layout()

	if save == True:
		plt.savefig(save_name)
	if show == True:	
		plt.show()


def plot_fnn(data, extent = [0.5,15.5, 14.5, 50.5], title ='false next neighbours', save = False, save_name = 'fnn_default00.png', R_threshold_series = True, R_threshold = 'undefined', start_order = 1, end_order = 15):
	"""
	:param data: A one dimensional float numpy array with the results of the false nearest neighbour algorithm for a fixed threshold as it can be obtained with 
		``eigval_tsa.param_opt.false_NN`` or a two dimensional float numpy array with the results of the false next neighbour algorithm for various thresholds as it 
		can be obtained with ``eigval_tsa.param_opt.various_R_threshold_fnn``.
	:param extent: A one dimensional array with the x and y limits of the colour map. The 0.5 offset is for a centered position of the ticks. 
	:param title: The string defines the title of the plot.
	:param save: If save is set to ``True``, the figure is saved. The default is ``False``.
	:param save_name: The string defines the name of the saved figure.
	:param R_threshold_series: This Boolean defines whether the data of which a plot shall be created is a two dimensional array of a threshold series or a one 
		dimensional array for a fixed threshold.
	:param R_threshold: This string defines the fixed threshold for a one dimensional time series.
	:param start_order: Defines the low limit dimension that is evaluated with the false next neighbour algorithm.
	:param end_order: Defines the upper limit dimension that is evaluated with the false next neighbour algorithm.
	:type data: One or two dimensional float numpy array.
	:type extent: One dimensional float array with four entries.
	:type title: String.
	:type save: Boolean.
	:type save_name: String.
	:type R_threshold_series: Boolean.
	:type R_threshold: String.
	:type start_order: Integer.
	:type end_order: Integer.
	:return: The result is a visualisation of the false next neighbour results with the desired title. It is possible to save the result.
	"""
	if R_threshold_series == True:
		fig = plt.figure()
		ax1 = fig.add_subplot(111)

		fnn = ax1.imshow(data[::-1,:], extent = [0.5,15.5, 14.5, 50.5], aspect = 'auto',cmap=cm.jet)
		ax1.set_xlabel('$d$', fontsize = 18 )
		ax1.set_ylabel('$R_{threshold}$', fontsize = 18 )
		cbar = plt.colorbar(fnn, extend='neither', spacing='proportional',
		                orientation='vertical', shrink=0.7)
		cbar.ax.tick_params(labelsize=15) 
		cbar.ax.set_title(r'$X_{\rm fnn}$', fontsize = 18)
		plt.xticks(fontsize =15)
		plt.yticks(fontsize =15)
		plt.title(title, fontsize = 18)
		plt.tight_layout()
		if save == True:
			plt.savefig(save_name)
		plt.show()
	if R_threshold_series == False:
		fig = plt.figure()
		plt.plot(np.arange(start_order, end_order + 1), data, 'b-', marker = 'v')
		plt.xlabel('$d$', fontsize = 18 )
		plt.ylabel(r'$X_{\rm fnn}$', fontsize = 18)
		plt.title(r'$R_{\rm threshold} =$' + R_threshold )
		if save == True:
			plt.savefig(save_name)
		plt.show()


def plot_avg_DD(data, start_lag = 1, end_lag = 10, lag_sampling = 1, label = 'undefined', lag_unit = 'sampling step', show_legend = True, title = 'avg distance from diagonal', save = False, save_name = 'avg_DD_default00.png'):
	"""
	:param data: A one dimensional float numpy array with the results of the function ``eigval_tsa.param_opt.avg_distance_from_diagonal``.
	:param start_lag: The first time lag that was evaluated with teh average distance from diagonal algorithm.
	:param end_lag: The last time lag that was evaluated with teh average distance from diagonal algorithm.
	:param lag_sampling: The sampling of the time delays that are evaluated with the average distance from diagonal algorithm.
	:param label: A string to define a name in the legend for the data results.
	:param lag_unit: A string that defines the units of the x-axis time lag :math:`\tau`.
	:param show_legend: If ``True``, the legend is shown. Otherwise not. Default is ``show_legend = True``.
	:param title: A string to set the title of the figure.
	:param save: If save is ``False`` as set by default, the image is not saved. Otherwise the image will be saved with the name defined by the string ``save_name``.
	:param save_name: A string that contains the name to save the plotted figure if ``save = True``.
	:type data: A one dimensional numpy array of floats.
	:type start_lag: Float.
	:type end_lag: Float.
	:type lag_sampling: Float.
	:type label: String.
	:type lag_unit: String.
	:type show_legend: Boolean.
	:type title: String.
	:type save: Boolean.
	:type save_name: String.
	:return: Plot of a figure with the results of the average distance from diagonal algorithm. The design is partly possible to choose and the figure can be optionally saved.
	"""
	x_axis_tau = np.arange(start_lag, end_lag + 1, lag_sampling)
	# print(x_axis_tau)
	ax = plt.subplot(111)
	ax.plot(x_axis_tau, data, 'bv-', label = label)
	#ax.plot(x_axis_tau, S_tau_henon_cutted, 'gv-', label = 'average distance from diagonal  \n(pre-bifurcation regime)') #### Nur fuer Henon!
	plt.xlim(0, end_lag)
	plt.xlabel(r'time lag $ \tau $ [' + lag_unit + ']', fontsize = 18)
	plt.ylabel(r'$\langle S\rangle$',fontsize = 18)
	plt.title(title, fontsize = 18)
	plt.xticks(fontsize = 15)
	plt.yticks(fontsize = 15)
	if show_legend:
		plt.legend(fontsize = 15)
	else:
		pass
	plt.tight_layout()
	if save == True:
		plt.savefig(save_name)
	plt.show()
