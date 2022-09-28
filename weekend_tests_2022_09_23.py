from main import create_waypoints, create_search_area, create_faux_tornado, create_damage_polygon, \
    create_waypoint_table, plotter
import random
import pandas as pd
from scipy.spatial.distance import squareform, pdist

from dynamic_routing import dynamic_route_with_init_route
from utilities import datetime_string


def tests(bounds=None, n_wpt=None, random_seed=None, plot=True, dynamic_test_to_do=None):
    if random_seed:
        random.seed(random_seed)
    if bounds:
        lb_x, ub_x, lb_y, ub_y = bounds
    else:
        lb_x, ub_x, lb_y, ub_y = 0, 10_000, 0, 10_000
    waypoints = create_waypoints(lb_x, ub_x, lb_y, ub_y, n_wpt, random_seed=random_seed)
    sbw, _verts = create_search_area(lb_x, ub_x, lb_y, ub_y)
    pt, direction, length, width = create_faux_tornado(sbw, lb_x, ub_x, lb_y, ub_y)
    damage = create_damage_polygon(pt, direction, length, width)
    waypoints_data = create_waypoint_table(waypoints, sbw, damage, 1)
    if plot:
        plotter(plot, waypoints, sbw, _verts, damage, pt,
                path=f"./weekend_tests_2022_09_26/t_{datetime_string(current=True)}_seed_{random_seed}_method_{dynamic_test_to_do}.png",
                title=f"Seed : {random_seed} | {datetime_string(current=True)}")
    dist_matrix = pd.DataFrame(squareform(pdist(waypoints)), columns=waypoints, index=waypoints)
    n_waypoints_to_visit, n_visit_first_uncover, n_visit_all_uncover = dynamic_route_with_init_route(waypoints,
                                                                                                     waypoints_data,
                                                                                                     dist_matrix,
                                                                                                     routing_mode=dynamic_test_to_do)
    kounter = 0
    while True:
        try:
            with open(file="weekend_tests_2022_09_26.csv", mode="a") as __f:
                __f.write(
                    f"{datetime_string(current=True)},{random_seed},{dynamic_test_to_do},"
                    f"{n_waypoints_to_visit},{n_visit_first_uncover},{n_visit_all_uncover}\n")
            break
        except:
            kounter += 1
            if kounter > 999:
                break
    if kounter >= 999:
        with open(file="weekend_tests_2022_09_26.csv", mode="a") as __f:
            __f.write(
                f"{datetime_string(current=True)},{random_seed},{dynamic_test_to_do},"
                f"{n_waypoints_to_visit},{n_visit_first_uncover},{n_visit_all_uncover}\n")


def make_random_seeds(n_tests, random_seed):
    if random_seed:
        random.seed(random_seed)
    seeds = set()
    while len(seeds) < n_tests:
        seeds.add(random.randint(1, 1_000_000))
    return list(seeds)


if __name__ == '__main__':
    random_seeds = make_random_seeds(n_tests=20, random_seed=8675309)
    dynamic_tests = ["do_nothing", "scores_in_order", "group_scores_in_order"]
    for seed in random_seeds:
        for dynamic_test in dynamic_tests:
            tests(bounds=(0, 5000, 0, 5000), n_wpt=1250, random_seed=seed, plot=True, dynamic_test_to_do=dynamic_test)
