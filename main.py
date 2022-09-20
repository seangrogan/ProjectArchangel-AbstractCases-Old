__base_output__ = "G:/project-archangel"

import random
from math import cos, sin, radians
from statistics import mean
from great_circle_calculator import great_circle_calculator
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import squareform, pdist
from shapely.geometry import Polygon, Point
from tqdm import tqdm

from dynamic_routing import dynamic_route_with_init_route
from utilities import datetime_string, automkdir


def main(bounds=None, n_wpt=None, random_seed=None, plot=True):
    if bounds:
        lb_x, ub_x, lb_y, ub_y = bounds
    else:
        lb_x, ub_x, lb_y, ub_y = 0, 10_000, 0, 10_000
    if random_seed:
        random.seed(random_seed)
    if n_wpt is None:
        n_wpt = 5_000
    waypoints = create_waypoints(lb_x, ub_x, lb_y, ub_y, n_wpt, random_seed=None)
    sbw, _verts = create_search_area(lb_x, ub_x, lb_y, ub_y)
    pt, direction, length, width = create_faux_tornado(sbw, lb_x, ub_x, lb_y, ub_y)
    damage = create_damage_polygon(pt, direction, length, width)

    waypoints_data = create_waypoint_table(waypoints, sbw, damage, 1)
    if plot:
        plotter(plot, waypoints, sbw, _verts, damage, pt)
    dist_matrix = pd.DataFrame(squareform(pdist(waypoints)), columns=waypoints, index=waypoints)
    dynamic_route_with_init_route(waypoints, waypoints_data, dist_matrix)


def create_waypoint_table(waypoints, sbw, damage, r_scan, default_score=0.5):
    def is_inside(_point, _geoms, _r_scan):
        if isinstance(_geoms, Polygon):
            _geoms = [_geoms]
        return any(geom.contains(Point(_point[0], _point[1])) for geom in _geoms) or any(
            geom.distance(Point(_point[0], _point[1])) <= _r_scan for geom in _geoms)

    data = {
        waypoint: {
            "damaged": is_inside(waypoint, damage, r_scan),
            "in_sbw": is_inside(waypoint, sbw, r_scan),
            "score": default_score * is_inside(waypoint, sbw, r_scan),
            "base_score" :  default_score * is_inside(waypoint, sbw, r_scan),
            "visited": False,
            "_wp":waypoint,
            "_wp_x":waypoint[0],
            "_wp_y": waypoint[1],
        }
        for waypoint in tqdm(waypoints)
    }
    return pd.DataFrame(data).transpose()


def plotter(plot, waypoints, sbw, _verts, damage, pt):
    if plot:
        x, y = zip(*waypoints)
        plt.scatter(x, y)
        x, y = sbw.exterior.xy
        plt.plot(x, y, color='red')
        x, y = zip(*_verts)
        plt.scatter(x, y, color='red')
        plt.scatter(pt[0], pt[1], color='red', marker='v')
        plt.scatter(x, y)
        x, y = damage.exterior.xy
        plt.plot(x, y, color='yellow')
        automkdir(f"./plots/test{datetime_string(current=True)}.png")
        plt.savefig(f"./plots/test{datetime_string(current=True)}.png")
        plt.show()
        plt.close()

    return waypoints, sbw, damage


def create_damage_polygon(pt, direction, length, width):
    if direction >= 0:
        direction = 90 - direction
    else:
        direction = 90 + abs(direction)
    x, y = pt
    p1 = (x + width * 0.5 * cos(radians(direction + 90)), y + width * 0.5 * sin(radians(direction + 90)))
    p2 = (x + width * 0.5 * cos(radians(direction - 90)), y + width * 0.5 * sin(radians(direction - 90)))
    pt_end = (x + length * cos(radians(direction)), y + length * sin(radians(direction)))
    x, y = pt_end
    p3 = (x + width * 0.5 * cos(radians(direction + 90)), y + width * 0.5 * sin(radians(direction + 90)))
    p4 = (x + width * 0.5 * cos(radians(direction - 90)), y + width * 0.5 * sin(radians(direction - 90)))
    damage = Polygon([[p1, p2, p3, p4][i] for i in ConvexHull([p1, p2, p3, p4]).vertices])
    return damage


def create_search_area(lb_x=0, ub_x=10_000, lb_y=0, ub_y=10_000,
                       min_area=0.33, max_area=0.67, n_vertex=4):
    d_x, d_y = ub_x - lb_x, ub_y - lb_y
    area_box = d_x * d_y
    p_bar = tqdm(desc="Creating Search Area")
    while True:
        vertices = set()
        while len(vertices) < n_vertex:
            pt = (random.randint(int(lb_x + .1 * d_x), int(ub_x - .1 * d_x)),
                  random.randint(int(lb_y + .1 * d_y), int(ub_y - .1 * d_y)))
            vertices.add(pt)
        vertices = list(vertices)
        hull = Polygon([vertices[i] for i in ConvexHull(list(vertices)).vertices])
        area_search_area = hull.area
        p_bar.set_postfix_str(f"{area_search_area / area_box:.3f} ({min_area} | {max_area})")
        p_bar.update()
        if min_area <= (area_search_area / area_box) <= max_area:
            break
    return hull, vertices


def create_faux_tornado(sbw, lb_x=0, ub_x=10_000, lb_y=0, ub_y=10_000):
    d_x, d_y = ub_x - lb_x, ub_y - lb_y
    p_bar = tqdm(desc="Generating Tornado Damage Area")
    min_x, min_y, max_x, max_y = sbw.bounds
    points = []
    while True:
        p_bar.update()
        x = random.randint(min_x, max_x)  # * random.randint(min_x, max_x), 0.5)
        y = random.randint(min_y, max_y)  # * random.randint(min_y, max_y), 0.5)
        pt = (x, y)
        if sbw.contains(Point(pt)):
            points.append(pt)
        if len(points) > 1000:
            del p_bar
            break
    x, y = zip(*points)
    pt = int(mean(x)), int(mean(y))
    tornado_track_file = pd.read_csv("1950-2021_all_tornadoes.csv")
    tornado_track_file = tornado_track_file[tornado_track_file["elat"] != 0]
    tornado_track_file['direction'] = tornado_track_file.apply(lambda row:
                                                               round(great_circle_calculator.bearing_at_p1(
                                                                   (row['slon'], row['slat']),
                                                                   (row['elon'], row['elat'])
                                                               )), axis=1)
    tornado_track_file = tornado_track_file[
        ~tornado_track_file['direction'].isin({-180, -135, -90, -45, 0, 45, 90, 135, 180})]

    directions = tornado_track_file['direction'].to_list()
    direction = random.choice(directions)

    tornado_track_file['len'] = tornado_track_file['len'] * 1609.344
    max_len = max(tornado_track_file['len'])
    lengths = [round(i * max(d_x, d_y) / max_len)
               for i in tqdm(tornado_track_file['len'].to_list(), desc="Getting Length")
               if round(i * max(d_x, d_y) / max_len)
               >= 0.1 * max(d_x, d_y)]

    length = random.choice(lengths)
    tornado_track_file['wid'] = tornado_track_file['wid'] * 0.9144
    max_wid = max(tornado_track_file['wid'])
    widths = [round(i * max(d_x, d_y) / max_wid)
              for i in tqdm(tornado_track_file['wid'].to_list(), desc="Getting Width")
              if length >= round(i * max(d_x, d_y) / max_wid)
              >= 0.05 * max(d_x, d_y)]
    width = random.choice(widths)
    return pt, direction, length, width


def create_waypoints(lb_x=0, ub_x=10_000, lb_y=0, ub_y=10_000, n_wpt=5_000, random_seed=None):
    waypoints = set()
    if random_seed:
        random.seed(random_seed)
    p_bar = tqdm(desc="Creating Waypoints")
    while len(waypoints) < n_wpt:
        pt = (random.randint(lb_x, ub_x), random.randint(lb_y, ub_y))
        waypoints.add(pt)
        p_bar.update()
    # print(len(waypoints))
    return list(waypoints)


if __name__ == '__main__':
    # main(random_seed=12345)
    # for i in [8675309, 19900305, 19890504]:
    main(bounds=(0, 5000, 0, 5000), n_wpt=1250, random_seed=8675309, plot=True)
