import copy
from collections import namedtuple
import concurrent.futures
import math

import numpy as np
import pandas as pd
import scipy.spatial.distance
from tqdm import tqdm
from utilities import euclidean


def dcf_function(d, scan_r, r=0.01, v=1):
    """v/(1+r)^(d/scan_r)
    """
    return min(v / pow((1 + r), (d / scan_r)), 1)


def one_over_e_to_the_x(d, scan_r, *args):
    return abs(1 - min(1 / math.exp((d / scan_r)), 1))


def one_over_log(d, scan_r, base=10, *args):
    return abs(1 - min(1 / math.log((d / scan_r)), 1))


def linear(d, scan_r, factor=1000):
    return abs(1 - min(d / (scan_r * factor), 1))


def update_scores(waypoint, waypoints_data, score_matrix, dist_matrix, lower):
    def update_score_matrix_function(distance, r_scan, function=dcf_function):
        return function(distance, r_scan)

    def update_waypoint_data_function(row, score_matrix, dist_matrix):
        if row.visited and row.damaged:
            return 1
        if row.visited and not row.damaged:
            return 0
        if not row.in_sbw:
            return 0



        # inv_dist_mat = dist_matrix[row._wp]
        inv_dist_mat = 10_000/dist_matrix[row._wp]
        inv_dist_mat.replace(np.inf, np.nan, inplace=True)
        inv_dist_mat.dropna(how="all", inplace=True)
        wt_mean = score_matrix.T[row._wp] / (inv_dist_mat * score_matrix.T[row._wp])
        return wt_mean.mean()
        # return score_matrix.T[row._wp].mean()

    _lower = 1 if lower else 0
    for wp2 in tqdm(waypoints_data["_wp"]):
        score_matrix.at[wp2, waypoint] = _lower - update_score_matrix_function(dist_matrix.at[waypoint, wp2], 25)
    # temp = pd.concat([score_matrix[waypoint].rename('score'), dist_matrix[waypoint].rename('dist')], axis=1)
    waypoints_data['score'] = waypoints_data.apply(
        lambda row: update_waypoint_data_function(row, score_matrix, dist_matrix), axis=1)
    return waypoints_data, score_matrix
    # score_matrix[waypoint] = score_matrix.apply(lambda column:
    #                                             _lower - update_function(dist_matrix.at[waypoint, wp2], 1)
    #                                             )


def update_scores_old(waypoints_data, dist_matrix, wp, lower):
    def update_fn(score, d, scan_r, reduce_score, visited, ub=500):
        fn = linear
        if visited:
            return score
        if d > ub:
            return score
        if reduce_score:
            return max(score - score * fn(d, scan_r) * 0.5, 0)
        else:
            return min(max(score + score * fn(d, scan_r) * 2, 0.1), 1)

    waypoints_data['score'] = waypoints_data.apply(
        lambda row: update_fn(
            row['score'], dist_matrix.at[wp, (row["_wp_x"], row["_wp_y"])], 1, lower, row['visited']
        ), axis=1)
    temp_df = waypoints_data
    temp_df['dist'] = waypoints_data.apply(
        lambda row: dist_matrix.at[wp, (row["_wp_x"], row["_wp_y"])]
        , axis=1)
    return waypoints_data


def dynamic_route_with_init_route(waypoints, waypoints_data, dist_matrix):
    init_waypoints_to_route = list(waypoints_data[waypoints_data['in_sbw'] == True].index)
    tour, dist = route_case(init_waypoints_to_route)
    score_matrix = pd.DataFrame(index=waypoints, columns=waypoints)
    score_matrix.fillna(0.5, inplace=True)
    for idx, wp in enumerate(tour):
        update_route = False
        waypoints_data.at[wp, "visited"] = True
        if waypoints_data.loc[[wp]].damaged.bool():
            waypoints_data.at[wp, 'score'] = 1
            waypoints_data, score_matrix = update_scores(wp, waypoints_data, score_matrix, dist_matrix, False)
            update_route = True
        else:
            waypoints_data.at[wp, 'score'] = 0
            waypoints_data, score_matrix = update_scores(wp, waypoints_data, score_matrix, dist_matrix, True)
        print(waypoints_data.loc[[wp]])
        if update_route:
            pass
    print(f"TADA")


def route_case(waypoints):
    waypoints = sorted(sorted(list(waypoints), key=lambda x: x[0]), key=lambda x: x[1])
    distance_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(waypoints))
    tour = construct_path_nearest_insertion_heuristic(distance_matrix, start_min_arc=True)
    route = [waypoints[idx] for idx in tour]
    dist = sum(euclidean(p1, p2)
               for p1, p2 in
               zip(route, route[1:]))
    return route, dist


def construct_path_nearest_insertion_heuristic(dist_matrix, start_min_arc):
    p_bar = tqdm(total=len(dist_matrix), position=0, leave=False, desc=f"Routing...")
    default_dist_matrix = copy.deepcopy(dist_matrix)
    D_ijk = namedtuple("D_ijk", ["i", "j", "k", "val"])
    n_cities = len(default_dist_matrix)
    if start_min_arc:
        for i in range(len(dist_matrix)):
            dist_matrix = _set_dist_mat_to(i, i, dist_matrix, val=float('inf'))
        desired_val = dist_matrix.min()
    else:
        desired_val = dist_matrix.max()
        for i in range(len(dist_matrix)):
            dist_matrix = _set_dist_mat_to(i, i, dist_matrix, val=float('inf'))
    __is, __js = np.where(desired_val == dist_matrix)
    __i, __j = int(__is[0]), int(__js[0])
    tour = [-999] + [__i, __j] + [-888]
    dist_matrix = _set_dist_mat_to(__i, __j, dist_matrix, val=float('inf'))
    while len(tour) < n_cities + 2:
        waypoint, dist_matrix, _other = _find_next_waypoint_to_insert(dist_matrix, tour)
        change_arc_list = [
            D_ijk(i, j, waypoint, _d_ijk(i, j, waypoint, default_dist_matrix))
            for i, j in zip(tour, tour[1:])
        ]
        change_arc_list.sort(key=lambda __x: __x.val)
        if change_arc_list:
            near_insert = change_arc_list.pop(0)
            while near_insert.k in tour:
                near_insert = change_arc_list.pop(0)
            idx_i = tour.index(near_insert.i)
            tour = tour[:idx_i + 1] + [near_insert.k] + tour[idx_i + 1:]
            for element in tour[1:-1]:
                dist_matrix = _set_dist_mat_to(near_insert.k, element, dist_matrix, val=float('inf'))
        else:
            assert False, "Something Has Gone Wrong Here!"
        p_bar.set_postfix_str(f"{len(tour) - 2}")
        p_bar.update()
    return tour[1:-1]


def _set_dist_mat_to(i, j, dm, val=float('inf')):
    dm[i, j] = val
    dm[j, i] = val
    return dm


def _get_val_from_dist_matrix(_i, _j, _dist_matrix):
    if _i in {-888, -999} or _j in {-888, -999}:
        return 0
    if _i == _j:
        return 0
    return _dist_matrix[_i, _j]


def _d_ijk(_i, _j, _k, _dist_matrix):
    return _get_val_from_dist_matrix(_i, _k, _dist_matrix) + \
           _get_val_from_dist_matrix(_k, _j, _dist_matrix) - \
           _get_val_from_dist_matrix(_i, _j, _dist_matrix)


def _find_next_waypoint_to_insert(_dist_matrix, tour):
    while True:
        min_val = _dist_matrix[tour[1:-1], :].min()
        _is, _js = np.where(min_val == _dist_matrix)
        _i, _j = int(_is[0]), int(_js[0])
        _dist_matrix = _set_dist_mat_to(_i, _j, _dist_matrix, val=float('inf'))
        if _i in tour and not (_j in tour):
            return _j, _dist_matrix, _i
        elif _j in tour and not (_i in tour):
            return _i, _dist_matrix, _j
        elif min_val >= float('inf'):
            assert False
