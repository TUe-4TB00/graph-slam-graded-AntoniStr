import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_landmark_measurement(graph, initial_estimate, result):
    pose_4 = initial_estimate.atPose2(X(4))
    landmark_2 = result.atPoint2(L(2))

    dx = float(landmark_2[0]) - pose_4.x()
    dy = float(landmark_2[1]) - pose_4.y()
    distance = math.hypot(dx, dy)
    rotation = math.degrees(math.atan2(dy, dx) - pose_4.theta())

    graph.add(gtsam.BearingRangeFactor2D(X(4), L(2), gtsam.Rot2.fromDegrees(rotation), distance, MEASUREMENT_NOISE))
    return graph