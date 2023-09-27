The "dominant_eigenvalue" Maths Guide
=====================================

In this chapter the main ideas of the methods and algorithms that are used in the ``eigval_tsa`` package are briefly summarized.

Takens' embedding theorem
--------------------------

Takens' theorem [Tak81]_ and its form generalized to noisy systems [Staetal97]_ is commonly used to investigate the complex dynamics of systems of which at least one time series is available. The theorem states that:

.. epigraph::
	Let M be a compact m dimensional manifold. If :math:`d \geq 2m+1`, then the set of :math:`(f_{\text{state}}, \varphi)` for which the map :math:`\Phi_{f_{\text{state}}, \varphi}(\xi)` is 	an embedding is open and dense in :math:`\mathcal{D}^r(M) \times \mathcal{C}^r(M,\mathbb{R})` for :math:`r\geq 1`, [Staetal97]_

with the system state :math:`\xi`, the measurement function :math:`\varphi : M \rightarrow \mathbb{R}` and the delay embedding map :math:`\Phi_{f_{\text{state}}, \varphi}: M \rightarrow \mathbb{R}^{\text{d}}` in the form :math:`\Phi_{f_{\text{state}}, \varphi}(\xi) = (\varphi (\xi), \varphi (f_{\text{state}}(\xi)), \varphi (f^2_{\text{state}}(\xi)), ..., \varphi (f^{(d-1)}_{\text{state}}(\xi)))`. An embedding is called an arbitrary map :math:`\Psi :M \rightarrow N` if it is diffeomorphic with respect to the manifolds :math:`M` and :math:`N`. The set of :math:`\mathcal{C}^r` diffeomorphisms of :math:`M` is denoted by :math:`\mathcal{D}^r(M)`. :math:`\mathcal{C}^r(M,\mathbb{R})` denotes the set of observation functions on :math:`M`. 

If the mathematical conditions of Takens' theorem are fulfilled, the mapped image :math:`\Psi (M)` is equivalent to :math:`M` itself and it is possible to define a map :math:`F_{\text{obs}} = \Psi \circ f_{\text{state}} \circ \Psi^{-1}` that will conserve all dynamical features of :math:`f_{\text{state}}` that are invariant under coordinate shifts. The important point is that the invariant features of :math:`F_{\text{obs}}` are accessible via the delay embedding variables

.. math::
	\vec{z}_{\text{n}} &= (\varphi (\xi_0), \varphi (f^{\text{n+1}}_{\text{state}}(\xi_0)), \varphi (f^{\text{n+2}}_{\text{state}}(\xi_0)), ..., \varphi (f^{\text{n+d-1}}_{\text{state}}(\xi_0))) \\
	&= (\varphi (\xi_0), \varphi (f^1_{\text{state}}(\xi_{\text{n}})), \varphi (f^2_{\text{state}}(\xi_{\text{n}})), ..., \varphi (f^{\text{d-1}}_{\text{state}}(\xi_{\text{n}})).
	

The map :math:`F_{\text{obs}}` gives the time evolution of the delay embedded variables:

.. math::
	F_{\text{obs}}(\vec{z}_{\text{n}}) &= \Phi \circ f_{\text{state}} \circ \Phi^{-1}(\vec{z}_{\text{n}}) = \Phi \circ f_{\text{state}} \circ \Phi^{-1}(\Phi (\xi_{\text{n}}) \\
	&= \Phi \circ f_{\text{state}}(\xi_{\text{n}}) = \Phi (\xi_{\text{n+1}}) = \vec{z}_{\text{n+1}} .

The following subsections give a short overview about to algorithm to choose a suitable embedding dimension and time delay to apply the delay embedding theorem. More detailed information about the Takens' embedding theorem can be found in [Hes18]_.

The false nearest neighbour algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to find a suitable time delay embedding dimension of a time series under consideration the false nearest neighbour algorithm is used as presented in [KS04]_ and [KBA92]_. The instructions of this chapter are used from [Hes18]_. 

Imagine a system of dimension :math:`m` that is tried to be embedded in an embedding space with the dimension :math:`d<m`. Some axes of the original state space are eliminated and points that are very different in their dynamics begin to lie next to each other in the delay embedding space. Such points are so-called false next neighbours because they would not be neighbours in a sufficient high delay embedding. Based on this idea it is necessary to evaluate

.. math::
	X_{\text{fnn}} = \frac{\sum_{n=1}^{N-m-1} \Theta \left( \frac{|s_n^{(m+1)}-s_{k(n)}^{(m+1)}|}{|s_n^{(m)}-s_{k(n)}^{(m)}|}-R_{\text{threshold}}\right)\Theta \left( \frac{\sigma}{R_{\text{threshold}}}-|s_n^{(m)}-s_{k(n)}^{(m)}|\right)}{\sum_{n=1}^{N-m-1} \Theta \left( \frac{\sigma}{R_{\text{threshold}}}-|s_n^{(m)}-s_{k(n)}^{(m)}|\right)}

with the standard deviation :math:`\sigma`, a distance threshold :math:`R_{\text{threshold}}`, the next neighbour :math:`s_{k(n)}^{(m)}` of :math:`s_n^{(m)}` in :math:`m` dimensions, whereas :math:`k(n)` denotes the index :math:`k` with :math:`|s_n-s_k| = \min` and the Heaviside function :math:`\Theta (\cdot )`. The threshold has to be chosen large in order to take into account exponential divergence due to deterministic chaos. The first Heaviside function in the nominator yields unity if the neighbours in :math:`m` dimensions are false next neighbours. In other words, their distance increases over the critical threshold :math:`R_{\text{threshold}}`. The second Heaviside function filters all pairs that cannot be false next neighbours due to an initial distance greater than :math:`\frac{\sigma}{R_{\text{threshold}}}`. The term defines the maximum possible distance on average. Depending on the threshold :math:`R_{\text{threshold}}` the ratio :math:`X_{\text{fnn}}` of false next neighbours should decrease with the embedding dimension :math:`m`. If a minimal value is reached, this is used as the right embedding dimension :math:`d` and the corresponding best time lag is calculated with the method of the following subsection. 


The average distance from diagonal algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The average distance from diagonal algorithm is introduced in [RCL94]_. Takens' embedding theorem does not require to choose a certain optimal time lag, but in practise it is necessary: If the time lag is small in comparison to the internal dynamics of the system, all delay embedded states lie relatively strong located to the main diagonal of the delay embedding space. If the time lag is very large, the delay embedded states fill the scatter in the whole delay embedding space and create a large cloud with the deterministic part closed in a very small structure. Therefore, the average distance :math:`\langle S_m\rangle` from diagonal that is defined by

.. math::
	\langle S_m\left(\tau \right)\rangle =\frac{1}{N}\sum_{i=1}^N |\vec{X}_i^\tau -\vec{X}_i^0| 

with the Euclidean distance measure :math:`|\cdot |` of the delay vector :math:`\vec{X}_i^\tau` from the non-delayed vector :math:`\vec{X}_i^0`, is considered. The formula can be written with the entries of a one dimensional time series :math:`X_t` by

.. math::
	\langle S_m\left(\tau \right)\rangle =\frac{1}{N}\sum_{i=1}^N \sqrt{\sum_{j=1}^{m-1} (X_{i+j\tau}-X_i)^2}

and is implemented in this way. More detailed informations can be found in [Hes18]_.


The eigenvalue estimation approach
-----------------------------------

The linear stability formalism of iterated maps is the starting point in order to derive a mathematical formalism to estimate the eigenvalues of the Jacobian of the linearized system that provide information about the stability of the system in a stable equilibrium. Without loss of generality the derivation is done for the two dimensional map

.. math::
	x_{\text{n+1}} &= f(x_{\text{n}}, y_{\text{n}}) \\
	y_{\text{n+1}} &= g(x_{\text{n}}, y_{\text{n}}).

A small perturbation :math:`\left\vert\eta^{\text{i}}_{\text{n}}\right\vert\ll 1` with :math:`i \in \lbrace x,y\rbrace` is applied to the fixed point :math:`(x^*,y^*)` and the functions :math:`f(x_{\text{n}}, y_{\text{n}})` and :math:`g(x_{\text{n}}, y_{\text{n}})` are expanded around the fixed point up to the linear order:

.. math::
	x^* + \eta^{\text{x}}_{\text{n+1}} &= f(x^*+\eta^{\text{x}}_{\text{n}}, y^*+\eta^{\text{y}}_{\text{n}}) \\
	&\approx \underbrace{f(x^*, y^*)}_{x^*} + \dfrac{\partial f}{\partial x}(x^*,y^*) \cdot \eta^{\text{x}}_{\text{n}} + \dfrac{\partial f}{\partial y}(x^*,y^*) \cdot \eta^{\text{y}}_{\text{n}} + \mathcal{O}\left(\left(\eta^{\text{x}}_{\text{n}}\right)^2,\left(\eta^{\text{y}}_{\text{n}}\right)^2,\eta^{\text{x}}_{\text{n}} \eta^{\text{y}}_{\text{n}}\right).

The result can be written with the Jacobian matrix :math:`J` evaluated at the fixed point :math:`(x^*,y^*)` and provides with

.. math::
	\vec{\eta}_{\text{n+1}} = \left( \begin{array}{c} \eta^{\text{x}}_{\text{n+1}} \\ 
	\eta^{\text{y}}_{\text{n+1}} \end{array} \right) = \left.\begin{pmatrix} \dfrac{\partial f}{\partial x} &  \dfrac{\partial f}{\partial y} \\
	\dfrac{\partial g}{\partial x} &  \dfrac{\partial g}{\partial y} \end{pmatrix}\right\vert_{(x^{\text{*}},y^{\text{*}})} \cdot \left( \begin{array}{c} \eta^{\text{x}}_{\text{n}} \\ 
	\eta^{\text{y}}_{\text{n}} \end{array} \right) = J\vert_{(x^{\text{*}},y^{\text{*}})} \cdot \vec{\eta}_{\text{n}}

the time evolution of the errors. The formula can be written as

.. math::
	\vec{\eta}'_{n+1} =\underline{\underline{\Lambda}}\cdot \vec{\eta}'_n

with the diagonalized Jacobian :math:`\underline{\underline{J}} = \underline{\underline{E}} \underline{\underline{\Lambda}}\underline{\underline{E}}^{-1}` with the diagonal matrix :math:`\underline{\underline{\Lambda}}` of eigenvalues and the matrix :math:`\underline{\underline{E}}` of right eigenvectors and the scaled perturbations :math:`\vec{\eta}' = \underline{\underline{E}}^{-1}\cdot \vec{\eta}`. Thus, the perturbation decays for eigenvalues :math:`|\lambda_{\text{i}}| < 1` and grows for :math:`|\lambda_{\text{i}}| > 1`. The formula can be expressed by time series values :math:`(x_n,y_n)` via the transformation :math:`\eta^{\text{x}}_{\text{n}} = x_{\text{n}} - x^*` and :math:`\eta^{\text{y}}_{\text{n}} =y_{\text{n}} -y^*` with 

.. math::
	\vec{x}_{\text{n+1}} - \vec{x}^* &= J\vert_{(x^{\text{*}},y^{\text{*}})} \cdot (\vec{x}_{\text{n}} - \vec{x}^*) \\
	\Leftrightarrow \vec{x}_{\text{n+1}} &= J\vert_{(x^{\text{*}},y^{\text{*}})} \cdot \vec{x}_{\text{n}}  \underbrace{-J\vert_{(x^{\text{*}},y^{\text{*}})} \vec{x}^*+ \vec{x}^*}_{\vec{x}_{\text{offset}}} \\
	\Leftrightarrow \vec{x}_{\text{n+1}} &= J\vert_{(x^{\text{*}},y^{\text{*}})} \cdot \vec{x}_{\text{n}} +\vec{x}_{\text{offset}} .

This formula can be explicitly written as 

.. math::
	\vec{z}_{\text{n+1}} &= \left( \begin{array}{c} x_{\text{n+1}} \\ x_{\text{n}} \end{array} \right) \\
 	&= \begin{pmatrix}
 	j_1 & j_2 \\
 	1 &  0 \end{pmatrix}
 	\left( \begin{array}{c} x_{\text{n}} \\x_{\text{n-1}} \end{array} \right) + \left( \begin{array}{c} z_{\text{offset}} \\ 0 \end{array} \right) \\ 
 	&= J_{\text{embed}} \cdot \vec{z}_{\text{n}} +\vec{z}_{\text{offset}}

with the two dimensional time delay embedding of a one dimensional time series. This formula is the numerical starting point for the implementation of the ``eigval_tsa.analysis.AR_EV_calc`` function. The coefficients :math:`c_1, c_2, ...` in the first row have to be estimated to get an expression for the Jacobian matrix :math:`J`. This estimation is done with an autoregression scheme in rolling windows. The scheme is briefly described in the following sub-section. More detailed information about the implementation of the eigenvalue estimation methods can be found in [Hes18]_.


The autoregression scheme
^^^^^^^^^^^^^^^^^^^^^^^^^^

The basic assumption of an autoregression scheme is that successive time series values depend on their :math:`p` past values in a simple linear manner described by

.. math::
	x_{\text{n}} = \phi_1 x_{\text{n-1}} +\phi_2 x_{\text{n-2}} + ... +\phi_p x_{\text{n-p}} + a_{\text{n}}

with an independent random shock :math:`a_{\text{n}}` for the n\textsuperscript{th} time step of a Gaussian white noise process and constant coefficients :math:`\phi_{1,2,...p}`. This can be written efficiently by 

.. math::
	\phi_{\text{p}} (\textbf{B})x_{\text{n}} = a_{\text{n}}

with the backshift operator 

.. math::
	\textbf{B}x_{\text{n}} = x_{n-1}

and the autoregressive operator

.. math::
	\phi (\textbf{B}) =  1-\phi_1 \mathbf{B}^1-\phi_2 \mathbf{B}^2-...-\phi_{\text{p}}\mathbf{B}^{\text{p}}.

There are :math:`p+2` unknown parameters :math:`\phi_n, \mu` and :math:`\sigma` that have to be estimated via a least squares fit of the time series data. AR(:math:`p`) models are widely used for stationary and nonstationary processes. [Boxetal15]_.

Bibliography
------------

.. [Tak81] Floris Takens. “Detecting strange attractors in turbulence”. In: Dynamical Systems and Turbulence, Warwick 1980. Ed. by David Rand and Lai-Sang Young. Berlin, Heidelberg: Springer Berlin Heidelberg, 1981, pp. 366–381. isbn: 978-3-540-38945-3.

.. [Staetal97] J. Stark et al. “Takens embedding theorems for forced and stochastic systems”. In: Nonlinear Analysis: Theory, Methods & Applications 30.8 (1997). Proceedings of the Second World Congress of Nonlinear Analysts, pp. 5303–5314. issn: 0362-546X. doi:https://doi.org/10.1016/S0362-546X(96)00149-6. url: http://www.sciencedirect.com/science/article/pii/S0362546X96001496.

.. [Hes18] Martin Heßler. “Leading indicators in B- and R-tipping systems with focus on eigenvalue estimation”. MA thesis. Westfälische Wilhelms-Universität Münster, 2018.

.. [KS04] Holger Kantz and Thomas Schreiber. Nonlinear Time Series Analysis. Second edition. Cambridge University Press, 2004. isbn: 0251 82150 9 hardback 0251 52902 6 paperback.



.. [RCL94] Michael T. Rosenstein, James J. Collins, and Carlo J. De Luca. “Reconstruction expansion as a geometry-based framework for choosing proper delay times”. In: Physica D: Nonlinear Phenomena 73.1 (1994), pp. 82–98. issn: 0167-2789. doi: https://doi.org/10.1016/0167-2789(94)90226-7. url: http://www.sciencedirect.com/science/article/pii/0167278994902267.

.. [Boxetal15] George E. P. Box et al. Time Series Analysis : Forecasting and Control.,Incorporated, 2015. ProQuest Ebook Central, https://ebookcentral.proquest.com/lib/ulbmuenster/detail.action?docID=2064681. Wiley Series in Probability and Statistics Ser. John Wiley & Sons, Incorporated, 2015. isbn: 9781118675021 (print) 9781118674918 (e-book). url: https://ebookcentral.proquest.com/lib/ulbmuenster/detail.action?docID=2064681.

