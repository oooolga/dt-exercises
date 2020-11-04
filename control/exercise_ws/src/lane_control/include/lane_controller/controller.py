import numpy as np
from duckietown.dtros import DTParam, ParamType
from copy import deepcopy
import sys

epsilon = 1e-10

class DummyLaneController:
    """
    The Lane Controller can be used to compute control commands from pose
    estimations.

    The control commands are in terms of linear and angular velocity (v, omega).
    The input are errors in the relative pose of the Duckiebot in the current
    lane.

    """

    def __init__(self, parameters):

        self.parameters = parameters

    def update_parameters(self, parameters):
        """Updates parameters of LaneController object.

            Args:
                parameters (:obj:`dict`): dictionary containing the new
                parameters for LaneController object.
        """
        self.parameters = parameters

    def get_car_control(self, **kwargs):
        """
        Returns a tuple of newly computed linear velocity and angular velocity
        """
        return self.parameters["~v_forward"].value, 0.0

class PurePursuitLaneController(DummyLaneController):

    def __init__(self, parameters):

        super(PurePursuitLaneController, self).__init__(parameters)

        self.half_speed_timer = 0.0
        self.half_speed_flag = False

        self.prev_v = 0.0, 0.0
        self.curr_k = self.parameters["~look_ahead_k"]

    def get_T_a_f_and_follow_point_robot(self, d, theta,
                                         v=None, k=None):
        T_ref_a = np.array([
               [np.cos(theta), -np.sin(theta), 0.],
               [np.sin(theta), np.cos(theta), d],
               [0., 0., 1.]
               ])

        if not v:
            v = self.parameters["~v_forward"].value

        if not k:
            k = self.curr_k

        look_ahead_d = k #* v
        
        fp_ref_x_square = np.maximum(0., look_ahead_d**2 - d**2)
        fp_ref_x = np.sqrt(fp_ref_x_square)

        T_ref_f = np.array([
               [1., 0., fp_ref_x],
               [0., 1., 0.],
               [0., 0., 1.]
               ])
        T_a_f = np.dot(np.linalg.inv(T_ref_a), T_ref_f)
        return T_a_f, np.array([T_a_f[0,2], T_a_f[1,2]])

    def get_car_control(self, **kwargs):

        # if not kwargs["in_lane"]:
        #     return self.prev_v[0], -self.prev_v[1]

        v_curr = v_init = self.parameters["~v_forward"].value
        self.curr_k = self.parameters["~look_ahead_k"]
        
        
        # if abs(kwargs["d_err"]) > 0.08:
        #     self.curr_k *= 0.5

        theta_err = kwargs["theta_err"]

        if abs(theta_err) > self.parameters["~slow_down_theta_thres"] or \
           abs(kwargs["d_err"]) > self.parameters["~slow_down_d_thres"]:

            v_curr = v_init * self.parameters["~slow_down_multiplier"]

        theta_err = np.clip(theta_err, -np.pi/2, np.pi/2)

        _, f_point = self.get_T_a_f_and_follow_point_robot(kwargs["d_err"],
                                                           theta_err,
                                                           v=v_curr)

        #d = np.sqrt(f_point[0]**2+f_point[1]**2)
        #alpha = np.arcsin(f_point[1] / d)
        alpha = np.arctan(f_point[1] / np.maximum(f_point[0], epsilon))

        self.prev_v = v_curr, alpha

        return self.prev_v



class BasicPIDLaneController(DummyLaneController):

    def __init__(self, parameters):

        # call parent class
        super(BasicPIDLaneController, self).__init__(parameters)

        # initialize parameters
        self.update_parameters(parameters)
        # compute k_d for critially damped if the flag is on
        if self.parameters["~k_d_critically_damped_flag"]:
            self.set_k_d_for_critically_damped()
        # compute d_thres for robutness
        if self.parameters["~d_threshold_robust_flag"]:
            self.set_d_threshold_for_robutness()

        # initialize terms for last error and accumulated errors
        self.prev_err = (0.0, 0.0) #(prev_d_err, prev_theta_err)
        self.I_err = (0.0, 0.0) #(I_d_err, I_tehta_err)

        # just in case if we want to normalize integral error terms
        self.time_elapsed = (0.0, 0.0) #(time_elapsed_d, time_elapsed_theta)

        # place holder for PID terms (debugging purposes)
        self.PID = [0.0]*6

    def _exchange_info(self, controllerB):

        self.prev_err = controllerB.prev_err
        self.I_err = controllerB.I_err

        # just in case if we want to normalize integral error terms
        self.time_elapsed = controllerB.time_elapsed

    def update_parameters(self, parameters):
        '''
        Update parameters.
        '''
        super(BasicPIDLaneController, self).update_parameters(parameters)

        if self.parameters["~k_d_critically_damped_flag"]:
            self.set_k_d_for_critically_damped()
        if self.parameters["~d_threshold_robust_flag"]:
            self.set_d_threshold_for_robutness()

    def get_car_control(self, **kwargs):
        err = (kwargs["d_err"], kwargs["theta_err"])
        # compute derivatives
        derr = self.compute_derivative_error(err)
        # compute integrals
        self.compute_and_set_I_error(err, kwargs["dt"],
                                     kwargs["wheels_cmd_flag"])
        # get constant forward moving speed and angular velocity
        RET = self.parameters["~v_forward"].value, \
              self.compute_angular_velocity(err, derr)
        self.set_prev_err(err)
        return RET

    def set_k_d_for_critically_damped(self):
        '''
        Calculate k_d's value based on k_theta and v_forward that satisfies the
        critically damped condition.
        '''
        k_d = -(self.parameters["~k_theta"].value ** 2 / \
               (4 * self.parameters["~v_forward"].value))
        self.parameters["~k_d"].set_value(k_d)

    def set_d_threshold_for_robutness(self):
        '''
        '''
        d_thres = abs(self.parameters["~k_theta"].value * \
                      self.parameters["~theta_threshold"] / \
                      self.parameters["~k_d"].value)
        self.parameters["~d_threshold"] = d_thres

    def compute_derivative_error(self, err):
        '''
        Compute the change in error terms.
        '''
        return err[0]-self.prev_err[0], err[1]-self.prev_err[1]

    def compute_and_set_I_error(self, err, dt, wheels_cmd_flag):
        if dt:
            self.I_err = self.I_err[0] + err[0]*dt, \
                         self.I_err[1] + err[1]*dt
            self.time_elapsed = self.time_elapsed[0]+dt, self.time_elapsed[1]+dt
            self.reset_I_err(err=err, wheels_cmd_flag=wheels_cmd_flag,
                             mode="if-needed")

    def set_prev_err(self, err):
        self.prev_err = err

    def reset_I_err(self, err=None, wheels_cmd_flag=None, mode="force-reset"):
        '''
        Force reset integral error terms or reset depending on conditions.
        '''
        assert mode in ["force-reset", "if-needed"]

        if mode == "hard-reset":
            self.I_err = (0.0, 0.0)
            self.time_elapsed = (0.0, 0.0)
        elif mode == "if-needed":
            assert err is not None
            assert wheels_cmd_flag is not None
            if not wheels_cmd_flag[0] and not wheels_cmd_flag[1]:
                self.I_err, self.time_elapsed = (0.0, 0.0), (0.0, 0.0)
                return
            # we reset integral error terms if current error is negligible
            # or there is change in direction of the errors
            if abs(err[0]) < self.parameters["~d_resolution"] or \
               np.sign(err[0]) != np.sign(self.prev_err[0]):
                self.I_err, self.time_elapsed = (0.0, self.I_err[1]), \
                                                (0.0, self.time_elapsed[1])
            if abs(err[1]) < self.parameters["~theta_resolution"] or \
               np.sign(err[1]) != np.sign(self.prev_err[1]):
                self.I_err, self.time_elapsed = (self.I_err[0], 0.0), \
                                                (self.time_elapsed[0], 0.0)

    def compute_angular_velocity(self, err, derr):
        '''
        Compute angular velocity for a PID controller that has constant
        velocity.
        '''
        self.PID = [self.parameters["~k_d"].value * \
                    np.clip(err[0], -self.parameters["~d_threshold"],
                                    self.parameters["~d_threshold"]), \
                    self.parameters["~k_theta"].value * \
                    np.clip(err[1], -self.parameters["~theta_threshold"],
                                    self.parameters["~theta_threshold"]), \
                    self.parameters["~k_Id"].value * self.I_err[0], \
                    self.parameters["~k_Itheta"].value * self.I_err[1], \
                    self.parameters["~k_dd"].value * derr[0], \
                    self.parameters["~k_dtheta"].value * derr[1]]

        return np.clip(sum(self.PID), -np.pi, np.pi)

    def get_parameters_str(self):
        RET = ""
        for key in self.parameters.keys():
            value = self.parameters[key].value \
                    if isinstance(self.parameters[key], DTParam) \
                    else self.parameters[key]

            RET += "{}: {}\n".format(key, value)
        return RET

    def get_PID_str(self):
        RET = "\tP:\n" + \
              "\t\td: {:.3f}".format(self.PID[0]) + \
              "\t\ttheta: {:.3f}\n".format(self.PID[1]) + \
              "\tI:\n" + \
              "\t\td: {:.3f}".format(self.PID[2]) + \
              "\t\ttheta: {:.3f}\n".format(self.PID[3]) + \
              "\tD:\n" + \
              "\t\td: {:.3f}".format(self.PID[4]) + \
              "\t\ttheta: {:.3f}\n".format(self.PID[5])
        return RET

class SpecialTurningPIDLaneController(DummyLaneController):

    '''
    SpecialTurningPIDLaneController class has two built-in PID controllers.
    One is for running in straight lines and one if for doing a turn.
    '''

    def __init__(self, parameters):
        # call parent class's __init__ function
        super(SpecialTurningPIDLaneController, self).__init__(parameters)

        # get turning controller's configuration
        turning_parameters = self.parameters.pop("~turning_config")
        self.turn_time_bound = turning_parameters.pop("forced_turn_time")
        self.half_v_criterion = turning_parameters.pop("half_speed_criterion")
        self.set_turning_parameters(turning_parameters)

        # forces straight line controller to be a PI controller
        self.parameters["~k_dd"].set_value(0.0)
        self.parameters["~k_dtheta"].set_value(0.0)

        # check values for turning controller
        assert self.turning_parameters["~k_dd"].value <= 0
        assert self.turning_parameters["~k_dtheta"].value <= 0

        # initialize both controllers
        self.controllers = {
                        "straight": BasicPIDLaneController(self.parameters),
                        "turn": BasicPIDLaneController(self.turning_parameters)
                        }

        # other settings
        self.curr_mode = "straight"
        self.prev_mode = "straight"
        self.turn_time_elapsed = 0.0
        self.half_v_flag = False
        self.prev_v = (self.turning_parameters["~v_forward"], 0.0)

    def set_turning_parameters(self, turning_parameters):
        '''
        Set up parameters for the turning controller.
        '''

        self.turning_parameters = deepcopy(self.parameters)

        for param in turning_parameters.keys():
            p = "~" + param

            if isinstance(self.turning_parameters[p], DTParam):
                self.turning_parameters[p].set_value(turning_parameters[param])
            else:
                self.turning_parameters[p] = turning_parameters[param]

    def get_car_control(self, **kwargs):

        if not kwargs["in_lane"]:
            return self.prev_v[0], -self.prev_v[1]

        err = (kwargs["d_err"], kwargs["theta_err"])

        if abs(err[1]) > self.turning_parameters["~theta_resolution"] or \
           bool(self.turn_time_elapsed > 0.0 and \
                self.turn_time_elapsed < self.turn_time_bound):
            # we think we are in a turn
            self.curr_mode = "turn"
            if kwargs["dt"]:
                self.turn_time_elapsed += kwargs["dt"]
                self.half_v_flag = True
        else:
            self.curr_mode = "straight"
            self.turn_time_elapsed = 0.0

        # if we changed controller, we need to exchange info
        if self.curr_mode != self.prev_mode:
            self.controllers[self.curr_mode]._exchange_info(
                                            self.controllers[self.prev_mode])

        RET = self.controllers[self.curr_mode].get_car_control(**kwargs)
        if self.curr_mode != self.prev_mode:
            RET2 = self.controllers[self.prev_mode].get_car_control(**kwargs)
            RET = 0.65*RET[0]+0.35*RET2[0], 0.65*RET[1]+0.35*RET2[1]

        # if yaw angle is too large, we reduce the speed
        if abs(err[1]) > self.half_v_criterion:
            RET = (RET[0]*0.5, RET[1])
            # self.turn_time_elapsed = sys.float_info.epsilon

        self.prev_mode = self.curr_mode
        self.prev_v = RET

        return RET

    def get_PID_str(self):
        return "[{} controller] PID info:\n".format(self.curr_mode.upper()) + \
                self.controllers[self.curr_mode].get_PID_str()

    def get_parameters_str(self):
        return "[{} controller] parameters:\n".format(self.curr_mode.upper()) + \
                self.controllers[self.curr_mode].get_parameters_str()
