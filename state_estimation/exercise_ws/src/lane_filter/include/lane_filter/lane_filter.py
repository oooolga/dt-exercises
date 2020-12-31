from collections import OrderedDict
from scipy.stats import multivariate_normal
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from math import floor, sqrt
from sklearn import linear_model
from sklearn.cluster import AgglomerativeClustering

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
        outliers = lm.outliers_
    else:
        m, b = float(lm.coef_), float(lm.intercept_)
        if mode == 'Huber':
            n_inlier = N - np.sum(lm.outliers_)
            outliers = lm.outliers_
        else:
            n_inlier = N
            outliers = np.zeros((N, 1), dtype=bool)

    return m, b, n_inlier, n_inlier/float(N), outliers

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
        self.fP = lambda F, P, L, Q: F @ P @ F.T + Q
        self.fK = lambda P, H, R: P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

        self.H = np.array([[1., 0.],
                           [0., 1.]])
        self.R = np.array([[0.1, 0],
                           [0, 0.1]])
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
        #self.F_k = self.fF(d_k_minus_1, phi_k_minus_1, dD_k, dphi_k)
        self.F_k = np.array([[1., 0.], [0., 1.]])
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
        separated_points = self.setup_segments_for_regression(pointsArray)
       
        white_inlier = yellow_inlier = total_inlier = 0
        white_m = yellow_m = 0.
        white_b = yellow_b = 0.
        white_inlier_propt = yellow_inlier_propt = 0
        
        if len(separated_points['YELLOW'][0]) >= 10:
            yellow_points = np.array([separated_points['YELLOW'][0],
                                     separated_points['YELLOW'][1]]).T
            yellow_cluster = self.point_clustering(yellow_points)

            best_cluster = None
            for cluster_i in range(yellow_cluster.n_clusters_):
                yellow_i = yellow_points[yellow_cluster.labels_ == cluster_i, :]
                cluster_i_center = np.mean(yellow_i, axis=0)
                cluster_i_dist = np.sum(np.power(cluster_i_center, 2))
                cluster_i_count = yellow_i.shape[0]

                if cluster_i_count < 10:
                    continue
                if best_cluster is None:
                    best_cluster = cluster_i, cluster_i_center, cluster_i_dist

                else:
                    best_cluster_i, best_cluster_center, best_cluster_dist = best_cluster

                    if best_cluster_dist>cluster_i_dist:
                        best_cluster = cluster_i, cluster_i_center, cluster_i_dist

            yellow_best_cluster = best_cluster
            if not best_cluster is None:
                yellow_regress_points = yellow_points[yellow_cluster.labels_ == best_cluster[0], :]
                yellow_m, yellow_b, yellow_inlier, yellow_inlier_propt, yellow_outliers = linearRegression(
                                                   yellow_regress_points[:,0][:, np.newaxis],
                                                   yellow_regress_points[:,1],
                                                   'Huber')
                yellow_inliers = yellow_regress_points[:,0][np.logical_not(yellow_outliers)], \
                                 yellow_regress_points[:,1][np.logical_not(yellow_outliers)]
                yellow_epsilon = yellow_inliers[1] - (yellow_m*yellow_inliers[0] + yellow_b)
            
        if len(separated_points['WHITE'][0]) >= 10:
            white_points = np.array([separated_points['WHITE'][0],
                                     separated_points['WHITE'][1]]).T
            white_cluster = self.point_clustering(white_points)

            best_cluster = None
            for cluster_i in range(white_cluster.n_clusters_):
                white_i = white_points[white_cluster.labels_ == cluster_i, :]
                cluster_i_center = np.mean(white_i, axis=0)
                cluster_i_dist = np.sum(np.power(cluster_i_center, 2))
                cluster_i_count = white_i.shape[0]

                if cluster_i_count < 10:
                    continue
                if best_cluster is None:
                    best_cluster = cluster_i, cluster_i_center, cluster_i_dist

                else:
                    best_cluster_i, best_cluster_center, best_cluster_dist = best_cluster

                    if best_cluster_dist>cluster_i_dist:
                        best_cluster = cluster_i, cluster_i_center, cluster_i_dist

            white_best_cluster = best_cluster
            if not best_cluster is None:
                white_regress_points = white_points[white_cluster.labels_ == best_cluster[0], :]
                white_m, white_b, white_inlier, white_inlier_propt, white_outliers = linearRegression(
                                                   white_regress_points[:,0][:, np.newaxis],
                                                   white_regress_points[:,1],
                                                   'Huber')
                white_inliers = white_regress_points[:,0][np.logical_not(white_outliers)], \
                                white_regress_points[:,1][np.logical_not(white_outliers)]
                white_epsilon = white_inliers[1] - (white_m*white_inliers[0] + white_b)
                

        white_z = np.array([0., 0.])
        yellow_z = np.array([0., 0.])
        if white_inlier:
            if white_b > 0.15:
                white_z = np.array([white_b - \
				    self.lanewidth + self.linewidth_white/2 - \
                                    (self.lanewidth-self.linewidth_white-self.linewidth_yellow)/2,
                                    -np.arctan(white_m*0.85)])
            else: 
                white_z = np.array([white_b+(self.lanewidth-self.linewidth_white)/2., \
		                    -np.arctan(white_m*0.85)])
            total_inlier += white_inlier
        if yellow_inlier:
            yellow_d = (self.lanewidth-self.linewidth_yellow)/2.-yellow_b
            yellow_z = np.array([yellow_d, -np.arctan(yellow_m*0.85)])
            total_inlier += yellow_inlier
        if total_inlier:
            if white_inlier and yellow_inlier:
                self.z = (white_z+yellow_z)/2.
            elif white_inlier:
                self.z = white_z
            else:
                self.z = yellow_z

        print(white_m, white_b, total_inlier, white_inlier, white_inlier_propt)
        print(yellow_m, yellow_b, total_inlier, yellow_inlier, yellow_inlier_propt)
        print(white_z, yellow_z, self.z)

        # TODO: Apply the update equations for the Kalman Filter to self.belief
        # d_variance = (np.sum(np.power(white_epsilon, 2)) + np.sum(np.power(yellow_epsilon, 2)))/(total_inlier-1)
        # R = np.array([[0., 0.], [0., d_variance]])
        residual_mean = self.z - self.H @ self.belief['mean']
        
        try:
            self.K_k = self.fK(self.belief['covariance'], self.H, self.R)
        except np.linalg.LinAlgError:
            self.K_k = np.array([[0., 0.], [0., 0.]])

        self.belief['covariance'] = self.belief['covariance'] - self.K_k @ self.H @ self.belief['covariance']
        self.belief['mean'] = self.z #self.belief['mean'] + self.K_k @ residual_mean
        

    def point_clustering(self, points, d_thres=0.13):
        clustering = AgglomerativeClustering(n_clusters=None,
                                             linkage='single',
                                             distance_threshold=d_thres).fit(points)
        return clustering

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
            #print(point.color, point.distance, point.pixel_ground.x, point.pixel_ground.y)
            # only consider points in a certain range from the Duckiebot for the position estimation
            if point.pixel_ground.x < self.d_max:
                if point.color == point.YELLOW and \
                   point.pixel_ground.y > self.edge_bound["YELLOW_Y"]["min"]:
                    pointsArray.append(point)
                if point.color == point.WHITE and \
                   point.pixel_ground.y > self.edge_bound["WHITE_Y"]["min"] and \
                   point.pixel_ground.y < self.edge_bound["WHITE_Y"]["max"]:
                    pointsArray.append(point)

        return pointsArray
