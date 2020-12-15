#!/usr/bin/env python3

import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import SegmentList, LanePose, BoolStamped, Twist2DStamped, FSMState, WheelEncoderStamped
from edge_point_msgs.msg import EdgePoint, EdgePointList
from lane_filter import LaneFilterHistogramKF
from sensor_msgs.msg import CompressedImage, Image
import os, cv2
import numpy as np
from cv_bridge import CvBridge


class LaneFilterNode(DTROS):
    """ Generates an estimate of the lane pose.

    Creates a `lane_filter` to get estimates on `d` and `phi`, the lateral and heading deviation from the center of the lane.
    It gets the segments extracted by the line_detector as input and output the lane pose estimate.


    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use

    Configuration:
        ~filter (:obj:`list`): A list of parameters for the lane pose estimation filter
        ~debug (:obj:`bool`): A parameter to enable/disable the publishing of debug topics and images

    Subscribers:
        ~segment_list (:obj:`SegmentList`): The detected line segments from the line detector
        ~car_cmd (:obj:`Twist2DStamped`): The car commands executed. Used for the predict step of the filter
        ~change_params (:obj:`String`): A topic to temporarily changes filter parameters for a finite time only
        ~switch (:obj:``BoolStamped): A topic to turn on and off the node. WARNING : to be replaced with a service call to the provided mother node switch service
        ~fsm_mode (:obj:`FSMState`): A topic to change the state of the node. WARNING : currently not implemented
        ~(left/right)_wheel_encoder_node/tick (:obj: `WheelEncoderStamped`): Information from the wheel encoders

    Publishers:
        ~lane_pose (:obj:`LanePose`): The computed lane pose estimate

    """

    def __init__(self, node_name):
        super(LaneFilterNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )

        veh = os.getenv("VEHICLE_NAME")

        self._filter = rospy.get_param('~lane_filter_histogram_kf_configuration', None)
        self._debug = rospy.get_param('~debug', False)
        self._predict_freq = rospy.get_param('~predict_frequency', 30.0)
        self._mode = rospy.get_param('~mode', None)

        self.loginfo('Current mode: {}'.format(self._mode))

        # Create the filter
        self.filter = LaneFilterHistogramKF(**self._filter)
        self.t_last_update = rospy.get_time()
        self.last_update_stamp = self.t_last_update

        self.filter.wheel_radius = rospy.get_param(f"/{veh}/kinematics_node/radius")
        self.filter.wheel_distance = rospy.get_param(f"/{veh}/kinematics_node/baseline")
        self.filter.wheel_trim = max(abs(rospy.get_param(f"/{veh}/kinematics_node/trim")), 0.1) #if self._mode != 'SIM' else 0.0
        self.filter.edge_bound = rospy.get_param("~edge_bound")


        # Subscribers
        self.sub_segment_list = rospy.Subscriber(f"/agent/ground_projection_node/lineseglist_out",
                                    EdgePointList,
                                    self.cbProcessSegments,
                                    queue_size=1)

        self.sub_encoder_left = rospy.Subscriber("~left_wheel_encoder_node/tick",
                                                 WheelEncoderStamped,
                                                 self.cbProcessLeftEncoder,
                                                 queue_size=1)

        self.sub_encoder_right = rospy.Subscriber("~right_wheel_encoder_node/tick",
                                                 WheelEncoderStamped,
                                                 self.cbProcessRightEncoder,
                                                 queue_size=1)

        self.sub_episode_start = rospy.Subscriber(f"episode_start",
                                                  BoolStamped,
                                                  self.cbEpisodeStart,
                                                  queue_size=1)


        # Publishers
        self.pub_lane_pose = rospy.Publisher("~lane_pose",
                                             LanePose,
                                             queue_size=1,
                                             dt_topic_type=TopicType.PERCEPTION)
        self.pub_ground_img = rospy.Publisher(f"~debug/edges/ground_compressed",
                                              CompressedImage,
                                              queue_size=1,
                                              dt_topic_type=TopicType.DEBUG)

        self.right_encoder_ticks = 0
        self.left_encoder_ticks = 0
        self.right_encoder_ticks_delta = 0
        self.left_encoder_ticks_delta = 0
        # Set up a timer for prediction (if we got encoder data) since that data can come very quickly
        rospy.Timer(rospy.Duration(1/self._predict_freq), self.cbPredict)
        self.ground_img_bg = None

        self.bridge = CvBridge()

    def cbEpisodeStart(self, msg):
        rospy.loginfo("Lane Filter Resetting")
        if msg.data:
            self.filter.reset()

    def cbProcessLeftEncoder(self, left_encoder_msg):
        if not self.filter.initialized:
            self.filter.encoder_resolution = left_encoder_msg.resolution
            self.filter.initialized = True
        self.left_encoder_ticks_delta = left_encoder_msg.data - self.left_encoder_ticks

    def cbProcessRightEncoder(self, right_encoder_msg):
        if not self.filter.initialized:
            self.filter.encoder_resolution = right_encoder_msg.resolution
            self.filter.initialized = True
        self.right_encoder_ticks_delta = right_encoder_msg.data - self.right_encoder_ticks

    def cbPredict(self,event):
        current_time = rospy.get_time()
        dt = current_time - self.t_last_update
        self.t_last_update = current_time

        # first let's check if we moved at all, if not abort
        if self.right_encoder_ticks_delta == 0 and self.left_encoder_ticks_delta == 0:
            return
        self.loginfo("[PREDICT]\n\tright_encoder_ticks_delta = {}\n".format(self.right_encoder_ticks_delta) +\
                                "\tleft_encoder_ticks_delta = {}".format(self.left_encoder_ticks_delta))

        self.filter.predict(dt, self.left_encoder_ticks_delta, self.right_encoder_ticks_delta)
        self.loginfo("{}".format(self.filter))        

        self.left_encoder_ticks += self.left_encoder_ticks_delta
        self.right_encoder_ticks += self.right_encoder_ticks_delta
        self.left_encoder_ticks_delta = 0
        self.right_encoder_ticks_delta = 0

        self.publishEstimate()


    def cbProcessSegments(self, edgepoint_list_msg):
        """Callback to process the segments

        Args:
            segment_list_msg (:obj:`SegmentList`): message containing list of processed segments

        """

        self.last_update_stamp = edgepoint_list_msg.header.stamp

        # Get actual timestamp for latency measurement
        timestamp_before_processing = rospy.Time.now()

        # Step 2: update
        self.filter.update(edgepoint_list_msg.points)
        self.loginfo("[UPDATE]\n{}\n".format(self.filter) +\
                     "\tmeasurement_z = {}".format(self.filter.z))
        ground_img = self.get_ground_img(edgepoint_list_msg)
        self.publishEstimate()

        debug_ground_img_msg = self.bridge.cv2_to_compressed_imgmsg(ground_img)
        debug_ground_img_msg.header = edgepoint_list_msg.header
        self.pub_ground_img.publish(debug_ground_img_msg)

    def get_ground_img(self, edgepoint_list_msg):
        if self.ground_img_bg is None:

            # initialize gray image
            self.ground_img_bg = np.ones((400, 400, 3), np.uint8) * 128

            # draw vertical lines of the grid
            for vline in np.arange(40,361,40):
                cv2.line(self.ground_img_bg,
                         pt1=(vline, 20),
                         pt2=(vline, 300),
                         color=(255, 255, 0),
                         thickness=1)

            # draw the coordinates
            cv2.putText(self.ground_img_bg, "-20cm", (120-25, 300+15), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)
            cv2.putText(self.ground_img_bg, "  0cm", (200-25, 300+15), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)
            cv2.putText(self.ground_img_bg, "+20cm", (280-25, 300+15), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)

            # draw horizontal lines of the grid
            for hline in np.arange(20, 301, 40):
                cv2.line(self.ground_img_bg,
                         pt1=(40, hline),
                         pt2=(360, hline),
                         color=(255, 255, 0),
                         thickness=1)

            # draw the coordinates
            cv2.putText(self.ground_img_bg, "20cm", (2, 220+3), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)
            cv2.putText(self.ground_img_bg, " 0cm", (2, 300+3), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)

            # draw robot marker at the center
            cv2.line(self.ground_img_bg,
                     pt1=(200 + 0, 300 - 20),
                     pt2=(200 + 0, 300 + 0),
                     color=(255, 0, 0),
                     thickness=1)

            cv2.line(self.ground_img_bg,
                     pt1=(200 + 20, 300 - 20),
                     pt2=(200 + 0, 300 + 0),
                     color=(255, 0, 0),
                     thickness=1)

            cv2.line(self.ground_img_bg,
                     pt1=(200 - 20, 300 - 20),
                     pt2=(200 + 0, 300 + 0),
                     color=(255, 0, 0),
                     thickness=1)

        # map segment color variables to BGR colors
        color_map = {EdgePoint.WHITE: (255, 255, 255),
                     EdgePoint.RED: (0, 0, 255),
                     EdgePoint.YELLOW: (0, 255, 255)}

        image = self.ground_img_bg.copy()

        # plot every segment if both ends are in the scope of the image (within 50cm from the origin)
        for point in edgepoint_list_msg.points:
            if not np.any(np.abs([point.pixel_ground.x, point.pixel_ground.y]) > 0.50):
                cv2.circle(image,
                           (int(point.pixel_ground.y * -400) + 200, int(point.pixel_ground.x * -400) + 300),
                           radius=0,
                           color=color_map.get(point.color, (0, 0, 0)),
                           thickness=-1)

        return image 


    def publishEstimate(self, segment_list_msg=None):

        belief = self.filter.getEstimate()

        # build lane pose message to send
        lanePose = LanePose()
        lanePose.header.stamp = self.last_update_stamp
        lanePose.d = belief['mean'][0]
        lanePose.phi = belief['mean'][1]
        lanePose.d_phi_covariance = [belief['covariance'][0][0],
                                     belief['covariance'][0][1],
                                     belief['covariance'][1][0],
                                     belief['covariance'][1][1]]
        lanePose.in_lane = True
        lanePose.status = lanePose.NORMAL

        self.pub_lane_pose.publish(lanePose)

    def cbMode(self, msg):
        return  # TODO adjust self.active

    def updateVelocity(self, twist_msg):
        self.currentVelocity = twist_msg

    def loginfo(self, s):
        rospy.loginfo('[%s] %s' % (self.node_name, s))


if __name__ == '__main__':
    lane_filter_node = LaneFilterNode(node_name="lane_filter_node")
    rospy.spin()
