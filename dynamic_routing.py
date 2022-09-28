import copy
from collections import namedtuple
import math

import numpy as np
import pandas as pd
import scipy.spatial.distance
from matplotlib import pyplot as plt
from tqdm import tqdm
from utilities import euclidean, datetime_string, automkdir


def update_scores(waypoints_data, influence_matrix, distance_matrix, influence_range):
    def update_waypoint_score(_row, _waypoints_data, _influence_matrix):
        if _row.visited and _row.damaged:
            return 1
        if _row.visited and not _row.damaged:
            return 0
        wp = _row["_wp"]
        score = _influence_matrix[wp].multiply(_waypoints_data['base_score']).sum() / _influence_matrix[wp].sum()
        return min(max(score, 0), 1)

    def update_waypoint_group_score(_row, _waypoints_data, _distance_matrix, _influence_range):
        if _row.visited and _row.damaged:
            return 1
        if _row.visited and not _row.damaged:
            return 0
        wp = _row["_wp"]
        nearby_wpts = list(_distance_matrix[wp][_distance_matrix[wp] <=_influence_range].index)
        score = _waypoints_data.loc[nearby_wpts]['score'].mean()
        return min(max(score, 0), 1)

    waypoints_data['score'] = waypoints_data.apply(
        lambda row: update_waypoint_score(row, waypoints_data, influence_matrix),
        axis=1
    )
    waypoints_data['group_score'] = waypoints_data.apply(
        lambda row: update_waypoint_group_score(row, waypoints_data, distance_matrix, influence_range),
        axis=1
    )
    return waypoints_data


def plot_stuff(waypoints_data, path, route=None):
    wpts_x, wpts_y, wpts_score = \
        waypoints_data._wp_x.to_list(), waypoints_data._wp_y.to_list(), waypoints_data.score.to_list()
    fig, ax = plt.subplots()
    ax.scatter(wpts_x, wpts_y, c=wpts_score, cmap="RdYlBu", vmin=0, vmax=1)
    if route:
        x, y = zip(*route)
        ax.plot(x, y)
    automkdir(path)
    plt.savefig(path)
    plt.close()


# def plot_stuff(waypoints_data, file_key, route=None):
#     p = Process(target=plot_the_stuff_for_mp, args=(waypoints_data, file_key, route))
#     p.start()


class InfluenceFunctions:
    @staticmethod
    def log(d, max_range, min_range=0):
        if d > max_range:
            return 0
        if d <= min_range:
            return 1
        _s = math.log(-1 * (d - min_range) + (max_range + 1), max_range)
        return max(min(_s, 1), 0)

    @staticmethod
    def linear(d, max_range, min_range=0):
        if d > max_range:
            return 0
        if d <= min_range:
            return 1
        _s = 1 - (d - min_range) / max_range
        return max(min(_s, 1), 0)

    @staticmethod
    def one_over_e_to_the_x(d, max_range, min_range=0):
        if d > max_range:
            return 0
        if d <= min_range:
            return 1
        _s = 1 / math.exp((d - min_range) / max_range)
        return max(min(_s, 1), 0)


def create_influence_matrix(waypoints, dist_matrix, max_range):
    influence_matrix = pd.DataFrame(index=waypoints, columns=waypoints)
    for wp1 in tqdm(waypoints):
        for wp2 in waypoints:
            influence_matrix.at[wp1, wp2] = InfluenceFunctions.log(
                dist_matrix.at[wp1, wp2], max_range)
    # x, y = zip(*waypoints)
    # mid_x = (max(x) - min(x)) / 2
    # mid_y = (max(y) - min(y)) / 2
    # middle = {wp: euclidean(wp, (mid_x, mid_y)) for wp in waypoints}
    # middle = min(middle, key=middle.get)
    # fig, ax = plt.subplots()
    # scrs = influence_matrix[middle].to_list()
    # ax.scatter(x, y, c=scrs, cmap="autumn", vmin=0, vmax=1)
    # ax.scatter(mid_x, mid_x, c='blue')
    # path = "./plots/infl_matrix_log.png"
    # automkdir(path)
    # plt.savefig(path)
    # plt.close()
    return influence_matrix


def update_route_function(init_tour, waypoints_data, dist_matrix,
                          route_as_visited, mode='do_nothing'):
    mode = mode.lower()
    new_tour = list()
    _wp = route_as_visited[-1]
    # waypoints_to_visit = waypoints_data[(waypoints_data['in_sbw'] == True) & (waypoints_data['visited'] == False)]
    waypoints_to_visit = waypoints_data[(waypoints_data['score'] > 0) & (waypoints_data['visited'] == False)]
    if mode in {'do_nothing'}:
        new_tour = init_tour[:]
    elif mode in {'order_scores', 'scores_in_order'}:
        new_tour = waypoints_to_visit.sort_values('score', ascending=False)["_wp"].to_list()
    elif mode in {'group_scores_in_order'}:
        new_tour = waypoints_to_visit.sort_values('group_score', ascending=False)["_wp"].to_list()
    elif mode in {'ni', 'nearest_insertion'}:
        new_tour, dist = route_case(waypoints_to_visit["_wp"].to_list())
    elif mode in {'dcf_by_dist', 'dcf_by_distance'}:
        ...  # todo
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented!")
    return new_tour


def dynamic_route_with_init_route(waypoints, waypoints_data, dist_matrix,
                                  influence_range=250, routing_mode='group_scores_in_order', notes=''):
    route_as_visited = []
    init_waypoints_to_route = list(waypoints_data[waypoints_data['in_sbw'] == True].index)
    tour, dist = route_case(init_waypoints_to_route)
    score_matrix = pd.DataFrame(index=waypoints, columns=waypoints)
    score_matrix.fillna(0.5, inplace=True)
    influence_matrix = create_influence_matrix(waypoints, dist_matrix, influence_range)
    n_waypoints_to_visit = len(waypoints_data[waypoints_data['in_sbw'] == True])
    idx, update_route, found = 0, False, False
    info1, fin1, info2, fin2 = "N/A 1", False, "N/A 2", False
    # plot_stuff(waypoints_data, f"./dynamic_route/{routing_mode}_with_route_{datetime_string()}_{notes}/route_{idx:05d}.png")
    # plot_stuff(waypoints_data, f"./dynamic_route/{routing_mode}_no_route_{datetime_string()}_{notes}/route_{idx:05d}.png")
    p_bar = tqdm(desc=f"Running Dynamic Program {routing_mode}", total=n_waypoints_to_visit)
    n_visit_first_uncover, n_visit_all_uncover = None, None
    while True:
        idx += 1
        if len(tour) <= 0:
            break
        if idx >= len(waypoints_data) - 1:
            break
        wp = tour.pop(0)
        waypoints_data.at[wp, "visited"] = True
        if waypoints_data.loc[[wp]].damaged.bool():
            waypoints_data.at[wp, 'score'] = 1
            waypoints_data.at[wp, 'base_score'] = 5
            update_route, found = True, True
        else:
            waypoints_data.at[wp, 'score'] = 0
            waypoints_data.at[wp, 'base_score'] = 0
            found = False
        waypoints_data = update_scores(waypoints_data,
                                       influence_matrix,
                                       dist_matrix,
                                       influence_range)

        # print(waypoints_data.loc[[wp]])
        route_as_visited.append(wp)
        # plot_stuff(waypoints_data,
        #            f"./dynamic_route/{routing_mode}_with_route_{datetime_string()}_{notes}/route_{idx:05d}.png",route=route_as_visited)
        # plot_stuff(waypoints_data,
        #            f"./dynamic_route/{routing_mode}_no_route_{datetime_string()}_{notes}/route_{idx:05d}.png")
        if update_route:
            if found:
                tour = update_route_function(tour, waypoints_data, dist_matrix, route_as_visited, mode=routing_mode)
            else:
                tour = update_route_function(tour, waypoints_data, dist_matrix, route_as_visited, mode=routing_mode)
        if not fin2 and len(waypoints_data[(waypoints_data.visited == True) & (waypoints_data.damaged == True)]) == \
                len(waypoints_data[(waypoints_data.damaged == True)]):
            info2 = f"Found all damaged points after {len(route_as_visited)} waypoints " \
                    f"vs an initial route of {n_waypoints_to_visit}"
            fin2 = True
            n_visit_all_uncover = len(route_as_visited)
        if not fin1 and len(waypoints_data[(waypoints_data.visited == True) & (waypoints_data.damaged == True)]) == 1:
            info1 = f"Found first damaged point after {len(route_as_visited)} waypoints " \
                    f"vs an initial route of {n_waypoints_to_visit}"
            fin1 = True
            n_visit_first_uncover = len(route_as_visited)
        p_bar.update()
    idx += 1
    # plot_stuff(waypoints_data, f"./dynamic_route/with_route_{datetime_string()}/route_{idx:05d}_fin.png",
    #            route=route_as_visited)
    # plot_stuff(waypoints_data, f"./dynamic_route/no_route_{datetime_string()}/route_{idx:05d}_fin.png")
    print(info1)
    print(info2)
    print(f"TADA")
    return n_waypoints_to_visit, n_visit_first_uncover, n_visit_all_uncover



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
