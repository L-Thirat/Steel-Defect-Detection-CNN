import pandas as pd
from os import listdir
from os.path import isfile, join

traindf = pd.read_csv('train.csv')

mypath = "train_images/"

onlyfiles = []
for f in listdir(mypath):
    if isfile(join(mypath, f)):
        if f in list(traindf["ImageId"]):
            selection = traindf[traindf["ImageId"]==f]["ClassId"]
            idx = list(selection.index)[0]
            for i in range(1, 5):
                if i not in list(selection):
                    traindf = traindf.append(traindf.loc[[idx] * 1].assign(ClassId=i,EncodedPixels=None), ignore_index=True)
        else:
            traindf = traindf.append(pd.DataFrame([[f, 1, None], [f, 2, None], [f, 3, None], [f, 4, None]], columns=["ImageId","ClassId","EncodedPixels"]), ignore_index=True)
traindf["ImageId_ClassId"] = traindf["ImageId"] + "_" + traindf["ClassId"].astype(str)
del traindf["ImageId"]
del traindf["ClassId"]
print(traindf)
print(traindf.shape)
traindf.to_csv('all_train_thirat.csv', index=False)
