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
    """
    The ``LineDetectorNode`` is responsible for detecting the line white, yellow and red line segment in an image and
    is used for lane localization.

    Upon receiving an image, this node reduces its resolution, cuts off the top part so that only the
    road-containing part of the image is left, extracts the white, red, and yellow segments and publishes them.
    The main functionality of this node is implemented in the :py:class:`line_detector.LineDetector` class.

    The performance of this node can be very sensitive to its configuration parameters. Therefore, it also provides a
    number of debug topics which can be used for fine-tuning these parameters. These configuration parameters can be
    changed dynamically while the node is running via ``rosparam set`` commands.

    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use

    Configuration:
        ~line_detector_parameters (:obj:`dict`): A dictionary with the parameters for the detector. The full list can be found in :py:class:`line_detector.LineDetector`.
        ~colors (:obj:`dict`): A dictionary of colors and color ranges to be detected in the image. The keys (color names) should match the ones in the Segment message definition, otherwise an exception will be thrown! See the ``config`` directory in the node code for the default ranges.
        ~img_size (:obj:`list` of ``int``): The desired downsized resolution of the image. Lower resolution would result in faster detection but lower performance, default is ``[120,160]``
        ~top_cutoff (:obj:`int`): The number of rows to be removed from the top of the image _after_ resizing, default is 40

    Subscriber:
        ~camera_node/image/compressed (:obj:`sensor_msgs.msg.CompressedImage`): The camera images
        ~anti_instagram_node/thresholds(:obj:`duckietown_msgs.msg.AntiInstagramThresholds`): The thresholds to do color correction

    Publishers:
        ~segment_list (:obj:`duckietown_msgs.msg.SegmentList`): A list of the detected segments. Each segment is an :obj:`duckietown_msgs.msg.Segment` message
        ~debug/segments/compressed (:obj:`sensor_msgs.msg.CompressedImage`): Debug topic with the segments drawn on the input image
        ~debug/edges/compressed (:obj:`sensor_msgs.msg.CompressedImage`): Debug topic with the Canny edges drawn on the input image
        ~debug/maps/compressed (:obj:`sensor_msgs.msg.CompressedImage`): Debug topic with the regions falling in each color range drawn on the input image
        ~debug/ranges_HS (:obj:`sensor_msgs.msg.Image`): Debug topic with a histogram of the colors in the input image and the color ranges, Hue-Saturation projection
        ~debug/ranges_SV (:obj:`sensor_msgs.msg.Image`): Debug topic with a histogram of the colors in the input image and the color ranges, Saturation-Value projection
        ~debug/ranges_HV (:obj:`sensor_msgs.msg.Image`): Debug topic with a histogram of the colors in the input image and the color ranges, Hue-Value projection

    """

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

        self.pub_d_color_edges = rospy.Publisher(
            "~debug/edges/color_compressed", CompressedImage, queue_size=1,
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

        # Fill in the segment_list with all the detected segments
        #segment_list = SegmentList()
        #segment_list.header.stamp = image_msg.header.stamp
        edge_pt_list = EdgePointList()
        edge_pt_list.header.stamp = image_msg.header.stamp

        for color, det in detections.items():
            # Get the ID for the color from the Segment msg definition
            # Throw and exception otherwise
            lines_normalized = (det+arr_cutoff) * arr_ratio
            color_id = getattr(Segment, color)
            edge_pt_l = self._to_segment_msg(lines_normalized, color_id, 40)
            #segment_list.segments.extend(segment_l)
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
                color_id = getattr(Segment, color)
                if color_id == 0:
                    color_edge[det[:,1], det[:,0], :] = 255
                if color_id == 1:
                    color_edge[det[:,1], det[:,0], :] = 255
                    color_edge[det[:,1], det[:,0], 0] = 0
            debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(color_edge)
            debug_image_msg.header = image_msg.header
            self.pub_d_edges.publish(debug_image_msg)


    @staticmethod
    def _to_segment_msg(coordinates, color, max_points=None):
        """
        Converts line detections to a list of Segment messages.

        Converts the resultant line segments and normals from the line detection to a list of Segment messages.

        Args:
            lines (:obj:`numpy array`): An ``Nx4`` array where each row represents a line.
            normals (:obj:`numpy array`): An ``Nx2`` array where each row represents the normal of a line.
            color (:obj:`str`): Color name string, should be one of the pre-defined in the Segment message definition.

        Returns:
            :obj:`list` of :obj:`duckietown_msgs.msg.Segment`: List of Segment messages

        """
        edge_pt_list = []
        
        #inrange_idx = []
        #segment = Segment()
        #for coordinate_i in range(coordinates.shape[0]):
        #    if color == segment.WHITE:
        #        if coordinates[coordinate_i, 0] < 0.3 and \
        #           coordinates[coordinate_i, 1] < 0.09 and \
        #           coordinates[coordinate_i, 1] > -0.3:
        #            inrange_idx.append(coordinate_i)
        #    if color == segment.YELLOW:
        #        if coordinates[coordinate_i, 0] < 0.3 and \
        #           coordinates[coordinate_i, 1] > -0.1:
        #            inrange_idx.append(coordinate_i)

        #if len(inrange_idx):
        #    coordinates = coordinates[np.array(inrange_idx), ::]
        #    N = coordinates.shape[0]
        #    if max_points and N > max_points:
        #        
        #        idx = np.arange(N)
        #        np.random.shuffle(idx)
        #        idx = idx[:max_points]
        #        coordinates = coordinates[idx,:]       

        for coordinate_i in range(0, coordinates.shape[0]):
            #segment = Segment()
            #segment.color = color
            #segment.pixels_normalized[0].x = coordinates[coordinate_i, 0]
            #segment.pixels_normalized[0].y = coordinates[coordinate_i, 1]
            #segment.pixels_normalized[1].x = coordinates[coordinate_i+1, 0]
            #segment.pixels_normalized[1].y = coordinates[coordinate_i+1, 1]
            #segment.normal.x = 0.
            #segment.normal.y = 0.
            #segment_msg_list.append(segment)

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
