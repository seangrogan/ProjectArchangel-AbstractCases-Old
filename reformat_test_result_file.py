import csv
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

data = pd.read_csv("weekend_tests_2022_09_23.csv")

outdata = defaultdict(dict)
for idx, row in tqdm(data.iterrows()):
    k = row['random_seed']
    t = row['dynamic_test_to_do']
    n = row['n_visit_all_uncover']
    outdata[t][k] = n

newdata = pd.DataFrame(outdata)  # , columns=("seed", "test", "n_to_all"), index=outdata.keys())

newdata.to_csv("BETTER_weekend_tests_2022_09_23.csv")