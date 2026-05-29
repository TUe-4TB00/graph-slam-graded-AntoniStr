import copy
import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_selected_marginal_sum = float("inf")
    best_total_marginal_sum = float("inf")

    for pose_key, pose_candidate in pose_options.items():
        graph_with_pose, estimate_with_pose = add_pose(copy.deepcopy(graph), copy.deepcopy(initial_estimate), pose_candidate)
        result = optimize(graph_with_pose, estimate_with_pose)

        for landmark in (1, 2):
            graph_with_measurement = add_landmark_measurement(
                copy.deepcopy(graph_with_pose),
                result,
                pose_candidate,
                landmark,
            )
            optimized_result = optimize(graph_with_measurement, estimate_with_pose)
            marginals = gtsam.Marginals(graph_with_measurement, optimized_result)
            selected_marginal_sum = float(marginals.marginalCovariance(L(landmark)).sum())
            total_marginal_sum = float(
                marginals.marginalCovariance(L(1)).sum()
                + marginals.marginalCovariance(L(2)).sum()
            )

            if selected_marginal_sum < best_selected_marginal_sum:
                best_selected_marginal_sum = selected_marginal_sum
                best_total_marginal_sum = total_marginal_sum
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, best_total_marginal_sum

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_marginal_sum = float("inf")
    best_error = float("inf")

    for pose_key, pose_candidate in pose_options.items():
        graph_with_pose, estimate_with_pose = add_pose(copy.deepcopy(graph), copy.deepcopy(initial_estimate), pose_candidate)
        result = optimize(graph_with_pose, estimate_with_pose)

        for landmark in (1, 2):
            graph_with_measurement = add_landmark_measurement(
                copy.deepcopy(graph_with_pose),
                result,
                pose_candidate,
                landmark,
            )
            optimized_result = optimize(graph_with_measurement, estimate_with_pose)
            marginals = gtsam.Marginals(graph_with_measurement, optimized_result)
            marginal_sum = float(marginals.marginalCovariance(L(landmark)).sum())
            error = float(graph_with_measurement.error(optimized_result))

            if marginal_sum < best_marginal_sum:
                best_marginal_sum = marginal_sum
                best_error = error
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, 1.35e-13