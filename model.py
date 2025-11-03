import pandas as pd
import numpy as np
import os
from PIL import Image

train_df = pd.read_csv("/kaggle/input/chexpert-v10-small/CheXpert-v1.0-small/train.csv")
train_df["Path"] = train_df["Path"].map(lambda path: os.path.join("/kaggle/input/chexpert-v10-small/", path))
train_df.head()

test_df = pd.read_csv("/kaggle/input/chexpert-v10-small/CheXpert-v1.0-small/valid.csv")
test_df["Path"] = test_df["Path"].map(lambda path: os.path.join("/kaggle/input/chexpert-v10-small/", path))
test_df.head()