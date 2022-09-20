import geopandas as gp
import pandas as pd
from tqdm import tqdm

from utilities import datetime_string


def _point_to_tuple(_point):
    return tuple([int(round(_point.x)), int(round(_point.y))])


infile = "G:/project-archangel-spatial-data/road-data/merged_points_102039.gpkg"
outfile = f"G:/project-archangel-spatial-data/road-data/points_of_interest_{datetime_string()}_crs_ESRI_102039.csv"

data = gp.read_file(infile, rows=100)

print(data.head())

out_data = {_point_to_tuple(item) for item in tqdm(data['geometry'])}

print("writing...")

df = pd.DataFrame(list(out_data), columns=['x', 'y'])

df.to_csv(outfile, index=False)
