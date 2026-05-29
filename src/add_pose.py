
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    dx = math.sqrt(2.0)
    dy = math.sqrt(2.0)
    dtheta = math.pi / 2.0

    odometry = gtsam.Pose2(dx, dy, dtheta)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))

    new_pose = gtsam.Pose2(4.0 + math.sqrt(2.0), math.sqrt(2.0), math.pi / 2.0)
    initial_estimate.insert(X(4), new_pose)

    return graph, initial_estimate