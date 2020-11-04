#!/usr/bin/env python3
import numpy as np
import rospy

# information regarding to DTROS/DTParam can be found in
# dt-ros-commons/packages/duckietown/include/duckietown/dtros
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, \
                                BoolStamped, FSMState, StopLineReading

from lane_controller.controller import DummyLaneController, \
                                       BasicPIDLaneController, \
                                       SpecialTurningPIDLaneController,\
                                       PurePursuitLaneController


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocitie.
    The configuration parameters can be changed dynamically while the node is
    running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that
        ROS will use
    Configuration:

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane
        filter
    """

    CONTROLLER_LOOKUP = {"dummy": DummyLaneController,
                         "basic_pid": BasicPIDLaneController,
                         "turn_pid": SpecialTurningPIDLaneController,
                         "pure_pursuit": PurePursuitLaneController}

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        ## Add the node parameters to the parameters dictionary (1) and
        ## initialize controller (2)
        # 1. get parameters
        self.params = dict()
        self.params["~v_forward"] = DTParam(
            "~v_forward",
            default=0.5,
            help="forward speed",
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=5.0
        )
        self.params["~k_theta"] = DTParam(
            "~k_theta",
            default=5.0,
            help="k_theta's value",
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params["~k_d"] = DTParam(
            "~k_d",
            help="k_d's value",
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params["~k_dd"] = DTParam(
            "~k_dd",
            default=5.0,
            help="k_dd's value",
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params["~k_dtheta"] = DTParam(
            "~k_dtheta",
            default=5.0,
            help="k_dtheta's value",
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params["~k_Id"] = DTParam(
            "~k_Id",
            default=0.0,
            help="k_Id's value",
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params["~k_Itheta"] = DTParam(
            "~k_Itheta",
            default=0.0,
            help="k_Id's value",
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params["~theta_threshold"] = rospy.get_param("~theta_threshold",
                                                          np.pi/6)
        self.params["~d_threshold"] = rospy.get_param("~d_threshold",
                                                      0.3)
        self.params["~look_ahead_k"] = rospy.get_param("~look_ahead_k",
                                                       3.0)
        self.params["~k_d_critically_damped_flag"] = rospy.get_param(
                                                "~k_d_critically_damped_flag",
                                                False
                                                )
        self.params["~d_resolution"] = rospy.get_param("~d_resolution",
                                                       0.005)
        self.params["~theta_resolution"] = rospy.get_param("~theta_resolution",
                                                           0.1)
        self.params["~d_threshold_robust_flag"] = rospy.get_param(
                                                "~d_threshold_robust_flag",
                                                False
                                                )

        self.params["~half_speed_time"] = rospy.get_param("~half_speed_time", 0.5)

        # 2. initialize controller
        self.controller_type = rospy.get_param("~controller_type", None)

        if self.controller_type == "turn_pid":
            self.params["~turning_config"] = rospy.get_param("~turning_config")

        assert self.controller_type in \
               LaneControllerNode.CONTROLLER_LOOKUP.keys()
        self.controller = LaneControllerNode.CONTROLLER_LOOKUP\
                            [self.controller_type](self.params)

        # initialize useful parameters
        self.last_time = None
        self.fsm_state = None
        self.wheels_cmd_executed = WheelsCmdStamped()
        self.stop_line_distance = None
        self.stop_line_detected = False
        self.at_stop_line = False

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose",
                                                 LanePose,
                                                 self.cbLanePoses,
                                                 queue_size=1)
        self.sub_lane_reading = rospy.Subscriber("~lane_filter",
                                                 LanePose,
                                                 self.cbLaneFilters,
                                                 queue_size=1)
        self.sub_intersection_navigation_pose = rospy.Subscriber(
                                                "~intersection_navigation_pose",
                                                LanePose,
                                                self.cbIntersectionPoses,
                                                queue_size=1
                                                )
        self.sub_wheels_cmd_execute = rospy.Subscriber("~wheels_cmd_executed",
                                                       WheelsCmdStamped,
                                                       self.cbWheelsCmdExecute,
                                                       queue_size=1)
        self.sub_fsm_node = rospy.Subscriber("~fsm_mode",
                                             FSMState,
                                             self.cbMode,
                                             queue_size=1)
        self.log("Initialized!")
        self.log("Lane controller type = {}.".format(self.controller_type))

    def cbMode(self, fsm_state_msg):
        self.fsm_state = fsm_state_msg.state
        # self.log("Current fsm state = {}.".format(self.fsm_state))

    def cbIntersectionPoses(self, input_pose_msg):
        self.cbLanePoses(input_pose_msg)

    def cbWheelsCmdExecute(self, msg_wheels_cmd):
        self.wheels_cmd_executed = msg_wheels_cmd

    def cbLaneFilters(self, input_pose_msg):
        self.log('[LANE FILTER] pose_msg = \n{}\n'.format(input_pose_msg))

    def cbLanePoses(self, input_pose_msg):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information
            about the current lane pose.
        """
        self.pose_msg = input_pose_msg

        # get time information
        duration = None
        if self.last_time:
            curr_time = rospy.Time.now().to_sec()
            duration = curr_time - self.last_time

        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.pose_msg.header

        # call controller's get control method to compute the car's next
        # time step's linear velocity and angular velocity
        wheels_cmd_flag = [self.wheels_cmd_executed.vel_left,
                           self.wheels_cmd_executed.vel_right]
        car_control_msg.v, car_control_msg.omega = \
            self.controller.get_car_control(d_err=self.pose_msg.d-self.pose_msg.d_ref,
                                            theta_err=self.pose_msg.phi-self.pose_msg.phi_ref,
                                            in_lane=self.pose_msg.in_lane,
                                            wheels_cmd_flag=wheels_cmd_flag,
                                            dt=duration)

        self.publishCmd(car_control_msg)

        # outputs pose information for current time step
        log_str = "\nfsm_state = {}\n".format(self.fsm_state) + \
                  "\nduration = {}s\n".format(duration) + \
                  "\npose_msg = \n{}\n".format(self.pose_msg) + \
                  "\ncar_control_msg = \n{}\n".format(car_control_msg) + \
                  "\nk = {}\n".format(self.controller.curr_k)
        if isinstance(self.controller, BasicPIDLaneController) or \
           isinstance(self.controller, SpecialTurningPIDLaneController):
            log_str = "\npid values = \n{}\n".format(
                                         self.controller.get_PID_str())
        self.log(log_str)
        self.last_time = rospy.Time.now().to_sec()


    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the
            requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)


    def cbParametersChanged(self):
        """Updates parameters in the controller object. (does not work??)"""

        self.controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
