#!/usr/bin/env python3

import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import Segment, SegmentList, AntiInstagramThresholds
from line_detector import LineDetector, ColorRange, plotSegments, plotMaps
from image_processing.anti_instagram import AntiInstagram
from edge_point_msgs.msg import EdgePoint, EdgePointList
from duckietown.dtros import DTROS, NodeType, TopicType


class LineDetectorNode(DTROS):

    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(LineDetectorNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )

        # Define parameters
        self._line_detector_parameters = rospy.get_param('~line_detector_parameters', None)
        self._colors = rospy.get_param('~colors', None)
        self._img_size = rospy.get_param('~img_size', None)
        self._top_cutoff = rospy.get_param('~top_cutoff', None)

        self.bridge = CvBridge()

        # The thresholds to be used for AntiInstagram color correction
        self.ai_thresholds_received = False
        self.anti_instagram_thresholds=dict()
        self.ai = AntiInstagram()

        # This holds the colormaps for the debug/ranges images after they are computed once
        self.colormaps = dict()

        # Create a new LineDetector object with the parameters from the Parameter Server / config file
        self.detector = LineDetector(**self._line_detector_parameters)
        # Update the color ranges objects
        self.color_ranges = {
            color: ColorRange.fromDict(d)
            for color, d in self._colors.items()
        }

        # Publishers
        self.pub_lines = rospy.Publisher(
            "~edge_point_list", EdgePointList, queue_size=1,
            dt_topic_type=TopicType.PERCEPTION
        )

        self.pub_d_edges = rospy.Publisher(
            "~debug/edges/compressed", CompressedImage, queue_size=1,
            dt_topic_type=TopicType.DEBUG
        )

        # these are not compressed because compression adds undesired blur
        
        # Subscribers
        self.sub_image = rospy.Subscriber(
            "~image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1
        )

        self.sub_thresholds = rospy.Subscriber(
            "~thresholds",
            AntiInstagramThresholds,
            self.thresholds_cb,
            queue_size=1
        )


    def thresholds_cb(self, thresh_msg):
        self.anti_instagram_thresholds["lower"] = thresh_msg.low
        self.anti_instagram_thresholds["higher"] = thresh_msg.high
        self.ai_thresholds_received = True


    def image_cb(self, image_msg):
        """
        Processes the incoming image messages.
        Performs the following steps for each incoming image:
        #. Performs color correction
        #. Resizes the image to the ``~img_size`` resolution
        #. Removes the top ``~top_cutoff`` rows in order to remove the part of the image that doesn't include the road
        #. Extracts the line segments in the image using :py:class:`line_detector.LineDetector`
        #. Converts the coordinates of detected segments to normalized ones
        #. Creates and publishes the resultant :obj:`duckietown_msgs.msg.SegmentList` message
        #. Creates and publishes debug images if there is a subscriber to the respective topics
        Args:
            image_msg (:obj:`sensor_msgs.msg.CompressedImage`): The receive image message
        """

        # Decode from compressed image with OpenCV
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr('Could not decode image: %s' % e)
            return


        # Perform color correction
        if self.ai_thresholds_received:
            image = self.ai.apply_color_balance(
                self.anti_instagram_thresholds["lower"],
                self.anti_instagram_thresholds["higher"],
                image
            )

        # Resize the image to the desired dimensions
        height_original, width_original = image.shape[0:2]
        img_size = (self._img_size[1], self._img_size[0])
        if img_size[0] != width_original or img_size[1] != height_original:
            image = cv2.resize(image, img_size, interpolation=cv2.INTER_NEAREST)
        image = image[self._top_cutoff:, :, :]


        # Extract the line segments for every color
        self.detector.setImage(image)
        detections = {
            color: self.detector.detectLines(ranges)
            for color, ranges in self.color_ranges.items()
        }

        # Remove the offset in coordinates coming from the removing of the top part and
        arr_cutoff = np.array([
            0, self._top_cutoff
        ])
        arr_ratio = np.array([
            1. / self._img_size[1], 1. / self._img_size[0]
        ])

        edge_pt_list = EdgePointList()
        edge_pt_list.header.stamp = image_msg.header.stamp

        for color, det in detections.items():
            lines_normalized = (det+arr_cutoff) * arr_ratio
            color_id = getattr(EdgePoint, color)
            edge_pt_l = self._to_coordinate_msg(lines_normalized, color_id, 40)
            edge_pt_list.points.extend(edge_pt_l)

        # Publish the message
        self.pub_lines.publish(edge_pt_list)

        # If there are any subscribers to the debug topics, generate a debug image and publish it
        if self.pub_d_edges.get_num_connections() > 0:
            color_edge = cv2.merge((
                                np.zeros(self.detector.canny_edges.shape, dtype=np.uint8),
                                np.zeros(self.detector.canny_edges.shape, dtype=np.uint8),
                                self.detector.canny_edges))
            for color, det in detections.items():
                color_id = getattr(EdgePoint, color)
                if color_id == 0:
                    color_edge[det[:,1], det[:,0], :] = 255
                if color_id == 1:
                    color_edge[det[:,1], det[:,0], :] = 255
                    color_edge[det[:,1], det[:,0], 0] = 0
            debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(color_edge)
            debug_image_msg.header = image_msg.header
            self.pub_d_edges.publish(debug_image_msg)


    @staticmethod
    def _to_coordinate_msg(coordinates, color, max_points=None):
        edge_pt_list = []     

        for coordinate_i in range(0, coordinates.shape[0]):

            pixel_pt = EdgePoint()
            pixel_pt.color = color
            pixel_pt.pixel_normalized.x = coordinates[coordinate_i, 0]
            pixel_pt.pixel_normalized.y = coordinates[coordinate_i, 1]
            edge_pt_list.append(pixel_pt)
        return edge_pt_list



if __name__ == '__main__':
    # Initialize the node
    line_detector_node = LineDetectorNode(node_name='line_detector_node')
    # Keep it spinning to keep the node alive
    rospy.spin()
