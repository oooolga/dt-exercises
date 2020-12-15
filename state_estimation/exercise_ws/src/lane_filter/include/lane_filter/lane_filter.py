
from collections import OrderedDict
from scipy.stats import multivariate_normal
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from math import floor, sqrt
from sklearn import linear_model

np.set_printoptions(precision=3)
EPS = 1e-5

def linearRegression(X, y, mode='Huber', max_points=None):

    N = X.shape[0]

    if max_points and N > max_points:
        idx = np.arange(N)
        np.random.shuffle(idx)
        idx = idx[:max_points]
        X = X[idx,:]
        y = y[idx]
        N = max_points
    
    if mode == 'RANSAC':
        lm = linear_model.RANSACRegressor(max_trials=200,
                                          stop_n_inliers=np.ceil(0.9*N))
    elif mode == 'Huber':
        lm = linear_model.HuberRegressor()
    else:
        lm = linear_model.LinearRegression()

    lm.fit(X, y)
    if mode == 'RANSAC':
        m, b = float(lm.estimator_.coef_), float(lm.estimator_.intercept_)
        n_inlier = np.sum(lm.inlier_mask_)
    else:
        m, b = float(lm.coef_), float(lm.intercept_)
        if mode == 'Huber':
            n_inlier = N - np.sum(lm.outliers_)
        else:
            n_inlier = N

    return m, b, n_inlier, n_inlier/float(N)

class LaneFilterHistogramKF():
    """ Generates an estimate of the lane pose.

    TODO: Fill in the details

    Args:
        configuration (:obj:`List`): A list of the parameters for the filter

    """

    def __init__(self, **kwargs):
        param_names = [
            # TODO all the parameters in the default.yaml should be listed here.
            'mean_d_0',
            'mean_phi_0',
            'sigma_d_0',
            'sigma_phi_0',
            'delta_d',
            'delta_phi',
            'd_max',
            'd_min',
            'phi_max',
            'phi_min',
            'cov_v',
            'linewidth_white',
            'linewidth_yellow',
            'lanewidth',
            'min_max',
            'sigma_d_mask',
            'sigma_phi_mask',
            'range_min',
            'range_est',
            'range_max',
            'error_offset'
        ]

        for p_name in param_names:
            assert p_name in kwargs
            setattr(self, p_name, kwargs[p_name])

        self.cov_mask = [self.sigma_d_mask, self.sigma_phi_mask]

        self.encoder_resolution = 0
        self.wheel_radius = 0.0
        self.wheel_distance = 0.0
        self.wheel_trim = 0.0
        self.initialized = False
        self.total_ticks = 135. # the number of ticks in one full revolution
        self.edge_bound = None

        self.f = lambda d, phi, dD, dphi: np.array([d+dD*np.sin(phi+dphi),
                                                    phi+dphi])
        self.fF = lambda d, phi, dD, dphi: np.array([(1, dD*np.cos(phi+dphi)),
                                                    (0, 1)])
        self.fL = lambda d, phi, dD, dphi: np.array([(np.sin(phi+dphi), dD*np.cos(phi+dphi)),
                                                    (0, 1)])
        self.fP = lambda F, P, L, Q: F @ P @ F.T + L @ Q @ L.T
        self.fK = lambda P, H, R: P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

        self.H = np.array([[1., 0.],
                           [0., 1.]])
        self.R = np.array([[0.03, 0],
                           [0, 0.07]])
        self.fQ = lambda eD, ephi: np.array([[eD**2+self.error_offset, eD*ephi],
                                             [eD*ephi, ephi**2+self.error_offset]])
        self.z = np.array([0., 0.])

        self.reset()


    def __str__(self):
        return "Estimated values:\n\t\td = {:.3f}\n\t\tphi = {}\n".format(self.belief['mean'][0],
                                                                    self.belief['mean'][1]) + \
               "Estimated variance:\n\t\tSigma =\n\t\t{}\n"\
					.format(str(self.belief['covariance']).replace('\n', '\n\t\t'))

    def reset(self):
        self.mean_0 = np.array([self.mean_d_0, self.mean_phi_0])
        self.cov_0 = np.array([[self.sigma_d_0, 0.],
                               [0., self.sigma_phi_0]])

        self.belief = {'mean': self.mean_0, 'covariance': self.cov_0}

    def predict(self, dt, left_encoder_delta, right_encoder_delta):
        #TODO update self.belief based on right and left encoder data + kinematics
        if not self.initialized:
            return

        # compute dD and dphi
        dD_left = 2*np.pi*self.wheel_radius*left_encoder_delta/self.encoder_resolution#/self.total_ticks
        dD_right = 2*np.pi*self.wheel_radius*right_encoder_delta/self.encoder_resolution#/self.total_ticks

        dD_k = (dD_left+dD_right) / 2.
        dphi_k = (dD_right-dD_left) / self.wheel_distance

        eD = dD_k/2.*abs(self.wheel_trim)
        ephi = np.sin(eD/abs(dD_k)) if abs(dD_k) > EPS else 0
        Q = self.fQ(eD, ephi)
        
        # compute predictions
        d_k_minus_1, phi_k_minus_1 = self.belief['mean']
        P_k_minus_1 = self.belief['covariance']
        self.belief['mean'] = self.f(d_k_minus_1, phi_k_minus_1, dD_k, dphi_k)
        self.F_k = self.fF(d_k_minus_1, phi_k_minus_1, dD_k, dphi_k)
        self.L_k = self.fL(d_k_minus_1, phi_k_minus_1, dD_k, dphi_k)
        self.belief['covariance'] = self.fP(self.F_k, P_k_minus_1, self.L_k, Q)

    def update(self, edgepoints):
        # prepare the segments for each belief array
        pointsArray = self.preparePoints(edgepoints)
        # generate all belief arrays

        ''' ## measurement likelihood approach
        measurement_likelihood = self.generate_measurement_likelihood(
            segmentsArray)

        # TODO: Parameterize the measurement likelihood as a Gaussian
        if np.sum(measurement_likelihood):
            d_margin = measurement_likelihood.sum(axis=1)
            phi_margin = measurement_likelihood.sum(axis=0)
            self.z = np.array([np.sum(np.mgrid[self.d_min:self.d_max+EPS:self.delta_d]*d_margin),
                               np.sum(np.mgrid[self.phi_min:self.phi_max+EPS:self.delta_phi]*phi_margin)])
       
        else: ## if no feasible edge is detected
            self.z = self.z # se to previous's z
            #self.z = np.array([0., 0.]) # for debugging purposes
            #residual_mean = np.array([0., 0.])

        
        '''
        separated_segments = self.setup_segments_for_regression(pointsArray)
       
        white_inlier = yellow_inlier = total_inlier = 0
        white_m = yellow_m = 0.
        white_b = yellow_b = 0.
        white_inlier_propt = yellow_inlier_propt = 0
        if len(separated_segments['WHITE'][0]) >= 10:
            white_m, white_b, white_inlier, white_inlier_propt = linearRegression(
                                               np.array(separated_segments['WHITE'][0])[:, np.newaxis],
                                               np.array(separated_segments['WHITE'][1]),
                                               30)
        if len(separated_segments['YELLOW'][0]) >= 8:
            yellow_m, yellow_b, yellow_inlier, yellow_inlier_propt = linearRegression(
                                               np.array(separated_segments['YELLOW'][0])[:, np.newaxis],
                                               np.array(separated_segments['YELLOW'][1]),
                                               30)
   
        white_z = np.array([0., 0.])
        yellow_z = np.array([0., 0.])
        if white_inlier:
            white_z = self.compute_regression_d_phi(white_m, white_b, mode='WHITE')
            total_inlier += white_inlier
        if yellow_inlier:
            yellow_z = self.compute_regression_d_phi(yellow_m, yellow_b, mode='YELLOW')
            total_inlier += yellow_inlier
        if total_inlier:
            self.z = (white_z * white_inlier + yellow_z * yellow_inlier) / total_inlier

        print(white_m, white_b, total_inlier, white_inlier, white_inlier_propt)
        print(yellow_m, yellow_b, total_inlier, yellow_inlier, yellow_inlier_propt)
        print(white_z, yellow_z, self.z)
        print(separated_segments)

        # TODO: Apply the update equations for the Kalman Filter to self.belief
        residual_mean = self.z - self.H @ self.belief['mean']
        
        self.K_k = self.fK(self.belief['covariance'], self.H, self.R)
        self.belief['covariance'] = self.belief['covariance'] - self.K_k @ self.H @ self.belief['covariance']
        self.belief['mean'] = self.z #self.belief['mean'] + self.K_k @ residual_mean
        

    def getEstimate(self):
        return self.belief

    def setup_segments_for_regression(self, segments, max_coordinate_points=None):

        separated_segments = {'WHITE':[[], []], 'YELLOW':[[],[]]}

        for segment in segments:

            if segment.color == segment.YELLOW:
                separated_segments['YELLOW'][0]+= [segment.pixel_ground.x]
                separated_segments['YELLOW'][1]+= [segment.pixel_ground.y]
        for segment in segments:
            if segment.color == segment.WHITE:
                separated_segments['WHITE'][0]+= [segment.pixel_ground.x]
                separated_segments['WHITE'][1]+= [segment.pixel_ground.y]
            
        return separated_segments

    def compute_regression_d_phi(self, m, b, mode):
        _p1 = np.array([0., b])
        _p2 = np.array([1., m+b])
        v12 = (_p2-_p1)/max(np.linalg.norm(_p2-_p1), EPS)
        n12 = np.array([-v12[1], v12[0]])
        d = np.inner(n12, _p1)
        phi = -np.arcsin(v12[1])

        if mode == 'WHITE':

            d = -d + self.linewidth_white/2.
            d = d - self.lanewidth/2.
        else:
            d = d - self.linewidth_yellow/2.
            d = self.lanewidth / 2 - d

        return np.array([d, phi])




    # generate a vote for one segment
    def generateVote(self, segment):
        p1 = np.array([segment.points[0].x, segment.points[0].y])
        p2 = np.array([segment.points[1].x, segment.points[1].y])
        t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)

        n_hat = np.array([-t_hat[1], t_hat[0]])
        d1 = np.inner(n_hat, p1)
        d2 = np.inner(n_hat, p2)
        l1 = np.inner(t_hat, p1)
        l2 = np.inner(t_hat, p2)
        if (l1 < 0):
            l1 = -l1
        if (l2 < 0):
            l2 = -l2

        l_i = (l1 + l2) / 2
        d_i = (d1 + d2) / 2
        phi_i = np.arcsin(t_hat[1])

        if segment.color == segment.WHITE:  # right lane is white
            if(p1[0] > p2[0]):  # right edge of white lane
                d_i = d_i - self.linewidth_white
            else:  # left edge of white lane

                d_i = - d_i

                phi_i = -phi_i
            d_i = d_i - self.lanewidth / 2

        elif segment.color == segment.YELLOW:  # left lane is yellow
            if (p2[0] > p1[0]):  # left edge of yellow lane
                d_i = d_i - self.linewidth_yellow
                phi_i = -phi_i
            else:  # right edge of white lane
                d_i = -d_i
            d_i = self.lanewidth / 2 - d_i

        # weight = distance
        weight = 1
        return d_i, phi_i, l_i, weight


    # prepare the segments for the creation of the belief arrays
    def preparePoints(self, points):
        pointsArray = []
        self.filtered_segments = []
        for point in points:

            # we don't care about RED ones for now
            if point.color != point.WHITE and point.color != point.YELLOW:
                continue
            # filter out any segments that are behind us
            if point.pixel_ground.x < 0:
                continue

            self.filtered_segments.append(point)
            # only consider points in a certain range from the Duckiebot for the position estimation
            if point.distance < self.d_max:
                if point.color == point.YELLOW and \
                   point.pixel_ground.y > self.edge_bound["YELLOW_Y"]["min"]:
                    pointsArray.append(point)
                if point.color == point.WHITE and \
                   point.pixel_ground.y > self.edge_bound["WHITE_Y"]["min"] and \
                   point.pixel_ground.y < self.edge_bound["WHITE_Y"]["max"]:
                    pointsArray.append(point)

        return pointsArray
