import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
import numpy as np

# load all training data
df = pd.read_csv('all_train_thirat.csv')
df["ImageId"] = df["ImageId_ClassId"].apply(lambda x: x.split("_")[0])

# split train - test
lst_select_test = list(set(df["ImageId"]))[:1800]
train = df[~(df['ImageId'].isin(lst_select_test))]
test = df[(df['ImageId'].isin(lst_select_test))]

del test["ImageId"]
del train["ImageId"]

# write train - test
train.to_csv('train_thirat.csv', index=False)
test.to_csv('test_thirat.csv', index=False)

# write submission file
test["EncodedPixels"] = None
test.to_csv('submission_thirat.csv', index=False)

# Move Test image
import os
import shutil

for file in lst_select_test:
    os.rename("train_images/%s" % file, "test_images/%s" % file)
