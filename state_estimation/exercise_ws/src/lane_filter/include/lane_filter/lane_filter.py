
from collections import OrderedDict
from scipy.stats import multivariate_normal
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from math import floor, sqrt

np.set_printoptions(precision=3)
EPS = 1e-5

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

        self.mean_0 = np.array([self.mean_d_0, self.mean_phi_0])
        self.cov_0 = np.array([[self.sigma_d_0, 0.],
                               [0., self.sigma_phi_0]])

        self.belief = {'mean': self.mean_0, 'covariance': self.cov_0}
        self.cov_mask = [self.sigma_d_mask, self.sigma_phi_mask]

        self.encoder_resolution = 0
        self.wheel_radius = 0.0
        self.wheel_distance = 0.0
        self.wheel_trim = 0.0
        self.initialized = False
        self.total_ticks = 135. # the number of ticks in one full revolution

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


    def __str__(self):
        return "Estimated values:\n\t\td = {:.3f}\n\t\tphi = {}\n".format(self.belief['mean'][0],
                                                                    self.belief['mean'][1]) + \
               "Estimated variance:\n\t\tSigma =\n\t\t{}\n"\
					.format(str(self.belief['covariance']).replace('\n', '\n\t\t'))


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

    def update(self, segments):
        # prepare the segments for each belief array
        segmentsArray = self.prepareSegments(segments)
        # generate all belief arrays

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

        residual_mean = self.z - self.H @ self.belief['mean']

        # TODO: Apply the update equations for the Kalman Filter to self.belief
        self.K_k = self.fK(self.belief['covariance'], self.H, self.R)
        self.belief['covariance'] = self.belief['covariance'] - self.K_k @ self.H @ self.belief['covariance']
        self.belief['mean'] = self.belief['mean'] + self.K_k @ residual_mean
        

    def getEstimate(self):
        return self.belief

    def generate_measurement_likelihood(self, segments):
        
        grid = np.mgrid[self.d_min:self.d_max+EPS:self.delta_d,
                        self.phi_min:self.phi_max+EPS:self.delta_phi]

        # initialize measurement likelihood to all zeros
        measurement_likelihood = np.zeros(grid[0].shape, dtype=np.float32)

        if len(segments) == 0:
            return measurement_likelihood

        for segment in segments:
            d_i, phi_i, l_i, weight = self.generateVote(segment)

            # if the vote lands outside of the histogram discard it
            if d_i > self.d_max or d_i < self.d_min or phi_i < self.phi_min or phi_i > self.phi_max:
                continue
            #d_i = np.clip(d_i, self.d_min, self.d_max)
            #phi_i = np.clip(phi_i, self.phi_min, self.phi_max)

            i = int(floor((d_i - self.d_min) / self.delta_d))
            j = int(floor((phi_i - self.phi_min) / self.delta_phi))
            measurement_likelihood[i, j] = measurement_likelihood[i, j] + 1

        if np.linalg.norm(measurement_likelihood) == 0:
            return measurement_likelihood

        # lastly normalize so that we have a valid probability density function

        measurement_likelihood = measurement_likelihood / \
            np.sum(measurement_likelihood)
        return measurement_likelihood




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

    def get_inlier_segments(self, segments, d_max, phi_max):
        inlier_segments = []
        for segment in segments:
            d_s, phi_s, l, w = self.generateVote(segment)
            if abs(d_s - d_max) < 3*self.delta_d and abs(phi_s - phi_max) < 3*self.delta_phi:
                inlier_segments.append(segment)
        return inlier_segments

    # get the distance from the center of the Duckiebot to the center point of a segment
    def getSegmentDistance(self, segment):
        x_c = (segment.points[0].x + segment.points[1].x) / 2
        y_c = (segment.points[0].y + segment.points[1].y) / 2
        return sqrt(x_c**2 + y_c**2)

    # prepare the segments for the creation of the belief arrays
    def prepareSegments(self, segments):
        segmentsArray = []
        self.filtered_segments = []
        for segment in segments:

            # we don't care about RED ones for now
            if segment.color != segment.WHITE and segment.color != segment.YELLOW:
                continue
            # filter out any segments that are behind us
            if segment.points[0].x < 0 or segment.points[1].x < 0:
                continue

            self.filtered_segments.append(segment)
            # only consider points in a certain range from the Duckiebot for the position estimation
            point_range = self.getSegmentDistance(segment)
            if point_range < self.range_est:
                segmentsArray.append(segment)

        return segmentsArray
