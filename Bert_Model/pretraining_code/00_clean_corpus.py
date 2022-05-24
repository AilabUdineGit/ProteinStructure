#!/usr/bin/env python
# coding: utf-8

from constants import *

import pandas as pd
import re
import os
from io import open

from tqdm import tqdm

# if not os.path.exists(CORPUS.all_clean_files):
#     os.makedirs(CORPUS.all_clean_files)

if not os.path.exists("data/train"):
    os.makedirs("data/train")
if not os.path.exists("data/test"):
    os.makedirs("data/test")

with open("../Ntrain_data.txt") as f:
    for idx, line in tqdm(enumerate(f)):
        line = " ".join(list(line.strip()))
        with open(f"data/train/{idx}.txt", "w") as out:
            out.write(line)

with open("../Ntest_data.txt") as f:
    for idx, line in tqdm(enumerate(f)):
        line = " ".join(list(line.strip()))
        with open(f"data/test/{idx}.txt", "w") as out:
            out.write(line)
