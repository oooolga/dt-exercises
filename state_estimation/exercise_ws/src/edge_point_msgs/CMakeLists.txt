cmake_minimum_required(VERSION 2.8.3)
project(edge_point_msgs)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  duckietown_msgs
  std_msgs
)

add_message_files(
  FILES
  EdgePoint.msg
  EdgePointList.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  duckietown_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime)

include_directories(${catkin_INCLUDE_DIRS})
