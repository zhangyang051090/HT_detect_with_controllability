import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

data_read = pd.read_csv("./data/RS232-T2000.csv",header=None,keep_default_na=False,lineterminator='\n',low_memory=False)

data_read.head(5)

#data_readd = data_read.iloc[:,[5,6]]

#data_ori = data_readd.values

#kmeans = KMeans(n_clusters=3, random_state=0).fit(data_ori)

data_read1 = data_read.iloc[:,[5]].values
data_read2 = data_read.iloc[:,[6]].values

data_read3 = data_read1/data_read2
data_read4 = data_read2/data_read1
#data_read5 = max(data_read3,data_read4)
#data_read5 = np.append(max(data_read3,data_read4),axis = 1)
data_read5 = np.append(data_read3,data_read4,axis = 1)
data_read6 = np.max(data_read5,axis = 1)
# print(data_read5)
# print(data_read6)

kmeans = KMeans(n_clusters=2, random_state=0).fit(data_read6.reshape(-1,1))




label_kmean = kmeans.labels_

print(label_kmean)

#data_read["c1"] = data_read3
#data_read["c0"] = data_read4
data_read["kmean"] = label_kmean
data_read["compare"] = data_read6

data_read.to_csv('./data/RS232-T2000_kmean.csv', encoding='utf-8-sig')