"""
This file turns points of interest into waypoints
"""
import pandas as pd
from tqdm import tqdm

from TornadoCaseGetter import TornadoCaseGetter
from TornadoCaseGetterV2ForPOItoWaypoints import TornadoCaseGetterV2ForPOItoWaypoints
from extra_project_utilities.waypoint_generation.hexagon import hexagon_pattern
from extra_project_utilities.waypoint_generation.retangular import rectangular
from extra_project_utilities.waypoint_generation.steiner_inspired.random_steiner_zone import random_steiner_zone
from extra_project_utilities.waypoint_generation.steiner_inspired.steiner_zone import steiner_zone
from extra_project_utilities.waypoint_generation.wpts_mean_shift import generate_waypoints_mean_shift_utm
from file_readers import read_sbw_file, _read_geo_files_into_geopandas
from pars.parfile_reader import parfile_reader
from utilities import automkdir, datetime_string


def _read_pointfile(pointfile):
    points = pd.read_csv(pointfile)
    points = points.to_records(index=False)
    points = list(tuple(pt) for pt in points)
    return points


def _create_waypoints(points, wpt_method, scanning_radius, pars):
    if not isinstance(wpt_method, str):
        results = dict()
        for method in wpt_method:
            results[method] = _create_waypoints(points, method, scanning_radius, pars)
        return results
    else:
        wpt_method = wpt_method.lower()
        if wpt_method in {"rectangular", "rec", "square", "rect"}:
            return rectangular(points, scanning_radius)
        elif wpt_method in {'hex', 'hexagon'}:
            return hexagon_pattern(points, pars, pars.get('flat_top', True))
        elif wpt_method in {'meanshift', 'mean_shift', 'mean shift', 'mean'}:
            return generate_waypoints_mean_shift_utm(points, scanning_radius)
        elif wpt_method in {'stiener', 'steiner_zone', 'steiner', "steiner zone"}:
            return steiner_zone(points, pars)
        elif wpt_method in {'random stiener', 'random_steiner_zone', 'random steiner', "random steiner zone"}:
            return random_steiner_zone(points, pars, improve_please=True)
        elif wpt_method in {'random stiener no improve', 'random_steiner_zone_no_improve'}:
            return random_steiner_zone(points, pars, improve_please=False)
        print(f"Method \"{wpt_method}\" not implemented")
        return set()


def create_waypoint_file(pointfile, scanning_r, parfile_name, limit_to_sbws=True):
    points = _read_pointfile(pointfile)
    print("==============================================")
    print(f"len of points now {len(points)}")
    print("==============================================")
    if limit_to_sbws:
        pars = parfile_reader(parfile_name)
        sbws = read_sbw_file(pars['sbws'], pars['crs'])
        damage_polygons = _read_geo_files_into_geopandas(pars['damage_polygons'], pars['crs'])
        tornado_cases = TornadoCaseGetterV2ForPOItoWaypoints(points, damage_polygons, sbws)
        dates = tornado_cases.dates[:]
        for date in tqdm(dates, desc="Getting Close Pts..."):
            points = set()
            _, tor_case, sbw = tornado_cases.get_specific_case(date)
            sbw = sbw.waypoints.to_list()
            for _s in sbw:
                points.update(set(_s))
            make_wpts(points, scanning_r, pars, date)
        points = list(points)
        print()
        print("==============================================")
        print(f"len of points now {len(points)}")
        print("==============================================")

def make_wpts(points, scanning_r, pars, date):
    methods = [
        "rectangular",
        "hexagon",
        "random_steiner_zone_no_improve",
        "random_steiner_zone",
        "mean_shift",
        "steiner_zone"
    ]
    for method in methods:
        print(f"Method '{method}' Scan r = {scanning_r}")
        wpts = _create_waypoints(points, method, scanning_r, pars)
        wpts = list(wpts)
        wpts.sort()
        df = pd.DataFrame(wpts)
        df = df.rename(columns={0: "EASTING", 1: "NORTHING"})
        outfile = f"G:/project-archangel-spatial-data/road-data/" \
                  f"_waypoint_data_expanded_by_date/{method}/" \
                  f"{date}_IN_SBW_waypoints_{scanning_r}m_nwpt_{len(wpts)}_{method}" \
                  f"T_{datetime_string()}" \
                  f"_crs_ESRI_102039.csv"
        automkdir(outfile)
        df.to_csv(outfile, index=False)
        print(f"\tCompleted Method '{method}'")


if __name__ == '__main__':
    pars = dict(
        pointfile="G:/project-archangel-spatial-data/road-data/"
                  "_poi_data_combined/points_of_interest_crs_ESRI_102039.csv",
        scanning_r=300,
        parfile_name="../pars/par2.json"
    )

    create_waypoint_file(
        **pars
    )
