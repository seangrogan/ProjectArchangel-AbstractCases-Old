import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
from tqdm import tqdm


def _read_file_fn(shp, keep=None):
    _gdf = gpd.read_file("zip://" + str(shp))
    if keep is not None:
        for k, v in keep.items():
            _gdf = _gdf[_gdf[k] == v]
    return _gdf


base_folder = "G:/wfo_data/" \
              "OUN_TSA_AMA_SHV_LZK_LCH_LIX_JAN_MEG_HUN_BMX_MOB_FFC_TAE_JAX_CHS_" \
              "PAH_OHX_MRX_LMK_JKL_20000101_20220615/wwa/"
out_folder = "G:/wfo_data/"

folder = Path(base_folder)
shapefiles = folder.glob("*.zip")
gdfs = []
p_bar = tqdm()
for shp in shapefiles:
    p_bar.set_postfix_str(str(os.path.split(shp)[1]))
    # gdf = _read_file_fn(shp) # for lsr
    gdf = _read_file_fn(shp, {"GTYPE": "P", "PHENOM": "TO"})
    if len(gdf) > 0:
        gdfs.append(gdf)
    p_bar.update()

print(f"loaded Shapefiles")
gdf = pd.concat(gdfs).pipe(gpd.GeoDataFrame)

print(f"concated Shapefiles")

gdf.to_file(f"G:/wfo_data/wwa_OUN_TSA_AMA_SHV_LZK_LCH_LIX_JAN_MEG_HUN_"
            f"BMX_MOB_FFC_TAE_JAX_CHS_PAH_OHX_MRX_LMK_JKL_20000101_20220615_Combined.shp.zip",
            driver="ESRI Shapefile")

print("Fin")
