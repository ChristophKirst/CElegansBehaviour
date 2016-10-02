# -*- coding: utf-8 -*-
"""
This module provides vairus Kalman filters to handle partial observations

Notes:
  Code is based on full observable kalman filters implemented in filterpy by Roger R Labbe Jr.
  http://github.com/rlabbe/filterpy
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'


from filterpy.common import dot3
from filterpy.kalman import unscented_transform
import filterpy.kalman.sigma_points as sigma_points

import numpy as np
from numpy import eye, zeros, dot, isscalar, outer
from scipy.linalg import inv, cholesky
from scipy.stats import multivariate_normal




class UnscentedKalmanFilter(object):
    """ Implements the Scaled Unscented Kalman filter (UKF) 
    
    Attributes:
      x (dim_x array): state vector
      P (dim_x x dim_x array): state covariance matrix
      R (dim_z x dim_z array): full observable measurement noise matrix assuming no occlusions
      Q (dim_x x dim_x array): process noise matrix

    Readable Attributes:
      K (array): Kalman gain from last update
      y (array): innovation
      likelihood (float):  likelihood of last measurement update
      log_likelihood (float): log likelihood of last measurement update.
      
    Notes:
      Nased on [1], using the formulation in [2]. 
      This filter scales the sigma points to avoid strong nonlinearities.
    
    References:
    .. [1] Julier, Simon J. "The scaled unscented transformation,"
           American Control Converence, 2002, pp 4555-4559, vol 6.
           https://www.cs.unc.edu/~welch/kalman/media/pdf/ACC02-IEEE1357.PDF

    .. [2] E. A. Wan and R. Van der Merwe, “The unscented Kalman filter for
           nonlinear estimation,” in Proc. Symp. Adaptive Syst. Signal
           Process., Commun. Contr., Lake Louise, AB, Canada, Oct. 2000
           https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf

    .. [3] S. Julier, J. Uhlmann, and H. Durrant-Whyte. "A new method for
           the nonlinear transformation of means and covariances in filters
           and estimators," IEEE Transactions on Automatic Control, 45(3),
           pp. 477-482 (March 2000).

    .. [4] Wan, Merle "The Unscented Kalman Filter," chapter in *Kalman
           Filtering and Neural Networks*, John Wiley & Sons, Inc., 2001.

    .. [5] R. Van der Merwe "Sigma-Point Kalman Filters for Probabilitic
           Inference in Dynamic State-Space Models" (Doctoral dissertation)          
    """

    def __init__(self, dim_x, fx, hx, points,
                 dt = 1.0, 
                 dim_z = None,
                 x_mean_fn = None, z_mean_fn = None,
                 sqrt_fn = cholesky,
                 x_residual = np.subtract, z_residual = np.subtract):
        """Constructor of a Kalman filter.

        Arguments:
          dim_x (int): number of state variables
          dim_z (int or None): maximal number of observatoin variables (None if not defined)
          dt (float): time step used in the underlying model
          hx (function(x,valid)): measurement function, transforms state into measurement, an optional valid bool array can specify the measurements that are considered to be valid
          points (class): sigma points, e.g. MerweScaledSigmaPoints or JulierSigmaPoints
          x_residual : (callable(x1, x2)), optional difference between states, optional for use e.g. if a state variable is circular         
          z_residual : (callable(z1, z2)), optional difference between full measurements,  optional for use e.g. if a state variable is circular
          x_mean_fn (callable(sigma_points, weights)), optional for use if a state variable lives on a circle etc. default = n.dot
          z_mean_fn (callable(sigma_points, weights)), optional for use if a measurement variable lives on a circle etc, default = n.dot
          sqrt_fn (callable(ndarray)), default = scipy.linalg.cholesky
        """

        self.Q = eye(dim_x)
        self.R = eye(dim_z)
        self.x = zeros(dim_x)
        self.P = eye(dim_x)
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.points_fn = points
        self.dt = dt
        self.num_sigmas = points.num_sigmas()
        self.hx = hx
        self.fx = fx
        self.x_residual = x_residual
        self.z_residual = z_residual
        self.x_mean = x_mean_fn
        self.z_mean = z_mean_fn
        self.log_likelihood = 0.0

        self.msqrt = sqrt_fn;
        
        # weights for the means and covariances.
        self.Wm, self.Wc = self.points_fn.weights()
        
        #cache
        self.sigmas_f = zeros((self.num_sigmas, self.dim_x))
        #self.sigmas_h = zeros((self._num_sigmas, self._dim_z))


    def predict(self, dt=None,  UT = unscented_transform, **fx_kwargs):
        """Performs the predict step of the UKF.

        Arguments:
          dt (double): time step, optional, default = self.dt
          UT (function(sigmas, Wm, Wc, noise_cov)): unscented transform
          fx_kwargs : optional keyword arguments to be passed into fx()
        """

        if dt is None:
            dt = self.dt
        
        # calculate sigma points for given mean and covariance
        sigmas = self.points_fn.sigma_points(self.x, self.P)
        print sigmas.shape
        
        for i in range(self.num_sigmas):
            self.sigmas_f[i] = self.fx(sigmas[i], dt, **fx_kwargs)

        self.x, self.P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q,
                            self.x_mean, self.x_residual)


    def update(self, z, valid = None,  R = None, UT = unscented_transform, **hx_kwargs):
        """ Update the UKF with the given measurements.
        
        Arguments:
          z (array): full measurement vector
          valid (dim_z array) optional array of bools indicating which of the measurments are valid
          R (array), optional measurement noise if different than internal one
          UT : function(sigmas, Wm, Wc, noise_cov), optional unscented transform
          hx_kwargs: optional keyword arguments to be passed into Hx
        """
        if z is None:
            return;

        if valid is None:
          valid = np.ones(self.dim_z, dtype = bool);
        dim_z = valid.sum(); #the current obs dim

        if R is None:
            R = self.R[np.ix_(valid,valid)];    
        elif isscalar(R):
            R = eye(dim_z) * R;

        self.sigmas_h = np.zeros((self.num_sigmas, dim_z));
        for i in range(self.num_sigmas):
            self.sigmas_h[i] = self.hx(self.sigmas_f[i], valid = valid, **hx_kwargs) #h assumed to return an array of the valid measurements only
        
        # adjust mean and residual calculation to valid measurments
        if self.z_mean is None:
          z_mean = None;
        else:
          def z_mean(sigmas, weights):
            full_sigmas = np.zeros((sigmas.shape[0], self.dim_z));
            full_sigmas[valid] = sigmas;
            full_mean = self.z_mean(full_sigmas, weights);
            return full_mean[valid];
        
        if self.z_residual is None:
          z_residual = None;
        else:
          def z_residual(x,y):
            full_x = np.zeros(self.dim_z); full_y = np.zeros(self.dim_z);
            full_x[valid] = x; full_y[valid] = y;
            full_res = self.z_residual(full_x, full_y);
            return full_res[valid];
        
        # mean and covariance of prediction passed through unscented transform
        zp, Pz = UT(self.sigmas_h, self.Wm, self.Wc, R, z_mean, z_residual);
        #zp = zp[valid];
        #Pz = Pz[np.ix_(valid,valid)];

        # compute cross variance of the state and the measurements
        Pxz = zeros((self.dim_x, dim_z))
        for i in range(self.num_sigmas):
            dx = self.x_residual(self.sigmas_f[i], self.x)
            dz = z_residual(self.sigmas_h[i], zp)
            Pxz += self.Wc[i] * outer(dx, dz)


        self.K = dot(Pxz, inv(Pz))        # Kalman gain
        self.y = z_residual(z[valid], zp)   # residual

        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot3(self.K, Pz, self.K.T)

        self.log_likelihood = multivariate_normal.logpdf(
            x=self.y, mean=np.zeros(len(self.y)), cov=Pz, allow_singular=True)


    @property
    def likelihood(self):
        return np.exp(self.log_likelihood);
