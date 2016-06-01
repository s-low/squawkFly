# squawkFly VUS (Very Ugly Submission Version)

3D Ball-Tracking and Trajectory Reconstruction from Fixed Stereo Views

Demo: https://www.youtube.com/watch?v=qbviYAziXVM

Given two simultaneous recordings of a ball-strike from fixed positions and some manually entered point correspondences between the views, the ball trajectory is extracted using a two dimensional kalman filter, interpolated between frames and synchronised. A full three-dimensional reconstruction is then built, using the epipolar geometry obtained with Hartley's Normalised eight-point algorithm.

Using the three-dimensional trajectory, and the known dimensions of the goalposts the ball-speed, curvature and distance travelled are estimated. The whole thing is automatically visualised with interactable 3D web graphics and the original videos are augmented with a ball 'trace' and annotated stats.
