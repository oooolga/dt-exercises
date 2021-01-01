# Duckietown Fall 2020 Exericse 3
## A Sneak Peak of the Final Results

### Right turns
[![Right Turn Results](https://img.youtube.com/vi/tlzEY17ByTM/0.jpg)](https://youtu.be/tlzEY17ByTM)

### Left turns
[![Left Turn Results](https://img.youtube.com/vi/RvWzFECeRoc/0.jpg)](https://youtu.be/RvWzFECeRoc)

## Overview
The objective of this exercise is to design a better estimator for lane pose measurements than Hough line segment detector. In duckietown, the control models rely on the visual inputs from the camera to generate steering commands for the duckiebots. In the default duckietown setting, Hough edge detector is utilized to identify lane segments and lane pose is estimated based on the orientations and positions of the lane segmets. Although the lane pose estimator using Hough features generates agreeable results, the estimator is not robust to image noises and performs poorly when the duckiebot deviates significantly from the mid-lane. The possible flaws in the Hough line segment detector includes:

* It relies heavily on the Hough's algorithm. If a pixel is identified as a point located on the lane but is not captured by the Hough segments, then the pixel would not be considered in the pose estimation process.
* Noise segments have great impacts on the estimation.
* An adequate line segment in the image generates same vote as any other segment (including noises) when the lane segments are aggregated. A long and accurate line segment detection, which may include many pixel points, is equally important as a short and mistaken segment, which may not contain many pixel points. In this voting system, pixels located on long segments are less important than pixels located on short segments.

In this exercise, a robust lane pose estimator is proposed such that it improves this baseline model in the following aspects:

* It does not rely on Hough's features.
* It is robust to outliers.
* Pixels detected on the lane are equally important.

The overview of this model is depicted in the following pseudo-code:

```
function get_lane_estimate(im):
   process im and obtain edges' pixel coordinates for each color lanes
   
   project the pixel on to the ground plane
   
   perform clustering on the lane coordinates for each pixel
   
   select a cluster for each color and run a robust regressor on every selected cluster
   
   obtain pose estimation based on the slope and intercept information of each color line
   
```


## Algorithm and Model Definition
### Image Preprocessing and Edge Filtering
The image is being preprocess by a Gaussian filter. The Canny edge detector is used to detect edges from the image.

### Pixel Clustering and Cluster Selection
Agglomerative clustering with single linkage was used to cluster the ground projected pixel coordinates. The algorithm groups the pixels based on their proximity to the other pixels in the cluster. For each color lane, the cluster whose the centroid is closest to the Duckiebot is being selected for regression.

The agglomerative clustering algorithm helps to detect the outliers in the lane pixels. Additionally, the agglomerative clustering algoirthm helps to filter out lanes if there are multiple lanes observed.

### Linear Regressor and Lane Pose Estimator

The Huber regressor is used to compute the slope $m$ and y-intercept $b$ of the lanes. The orientation angle, $\theta$, is computed as 
\begin{equation}
\theta = \arctan(m)
\end{equation}
and the distance $d$ from the midlane can be computed as
\begin{equation}
d = b_{\text{white}} + (\text{lane_width}-\text{white_line_width})/2
\end{equation}
or 
\begin{equation}
d = (\text{lane_width}-\text{yellow_line_width})/2-b_{\text{yellow}}
\end{equation}


## Results

The simulation results for left and right turns are shown in Section `A Sneak Peak of the Final Results`. This model is able to render in the Duckietown simulator for over 10 minutes without crashing.
