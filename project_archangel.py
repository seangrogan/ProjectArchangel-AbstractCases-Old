# the aim of this project is to dynamically
# route vehicles in the event of a tornado case
import argparse

from tqdm import tqdm

from TornadoCaseGetter import TornadoCaseGetter
from file_readers import read_waypoint_file, read_sbw_file, _read_geo_files_into_geopandas
from pars.parfile_reader import parfile_reader


def main(
        parfile_name
):
    pars = parfile_reader(parfile_name)
    waypoints = read_waypoint_file(pars['waypoints'])
    sbws = read_sbw_file(pars['sbws'], pars['crs'])
    damage_polygons = _read_geo_files_into_geopandas(pars['damage_polygons'], pars['crs'])
    tornado_cases = TornadoCaseGetter(waypoints, damage_polygons, sbws)
    dates = tornado_cases.dates[:]
    print(len(dates))
    for date in tqdm(dates, desc="Plotting..."):
        tornado_cases.plot_polys(date, out_path='./plots_part_crs_102039/')
    pass


def arg_parser():
    parser = argparse.ArgumentParser(description='Project Archangel')
    parser.add_argument('-parfile_name', '-parfile', '-par', '-pars', '-p', action='store')
    parser.add_argument('-random_seed', action='store', default=None, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    _args = arg_parser()
    main(
        _args.parfile_name
    )
