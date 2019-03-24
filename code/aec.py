#_______________________________________________________________________________
# CE888 Project |        aec.py       | Ogulcan Ozer. | Feb. 2019 | UNFINISHED.
#_______________________________________________________________________________

##************************
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
from sklearn import cluster, datasets, metrics
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from data import data_struct as DS
##************************

path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))
#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def read_data():

    #Read and seperate the mnist dataset.
    mnist = input_data.read_data_sets(os.path.join(path,'mnist'))
    tr = mnist.train.images
    trl = mnist.train.labels
    te = mnist.test.images
    tel = mnist.test.labels

    mnist_data = DS(tr,trl,te,tel)

    #Read and seperate the HAR dataset.
    tr1 = pd.read_csv(os.path.join(path,'UCI HAR Dataset','train',
    'X_train.txt'),delim_whitespace=True,error_bad_lines=False, warn_bad_lines=False,
    header=None)
    trl1 = pd.read_csv(os.path.join(path,'UCI HAR Dataset','train',
    'y_train.txt'), error_bad_lines=False, warn_bad_lines=False,header=None)
    te1 = pd.read_csv(os.path.join(path,'UCI HAR Dataset','test',
    'X_test.txt'),delim_whitespace=True,error_bad_lines=False, warn_bad_lines=False,
    header=None)
    tel1 = pd.read_csv(os.path.join(path,'UCI HAR Dataset','test',
    'y_test.txt'),error_bad_lines=False, warn_bad_lines=False, header=None)
    har_data = DS(tr1,trl1.values.T[0],te1,tel1.values.T[0])

    #Read and seperate the pulsar dataset.
    alldata = pd.read_csv(os.path.join(path,'HTRU2\HTRU_2.csv'),
    sep=',',error_bad_lines=False, warn_bad_lines=False,header=None)
    labels = pd.DataFrame(alldata[alldata.columns[8]])
    alldata.drop(alldata.columns[8], axis = 1, inplace = True)
    tr2, te2, trl2, tel2 = train_test_split(alldata, labels, test_size=0.20,
    random_state=42)
  
    pulsar_data = DS(tr2,trl2.values.T[0],te2,tel2.values.T[0])

    return mnist_data, har_data, pulsar_data

def cluster_data(data, labels, cluster_size):

    # kmeans clustering 
    kM=cluster.KMeans(n_clusters=cluster_size)
    kM.fit(data)
    kM.labels_
    hg, cp, v = metrics.homogeneity_completeness_v_measure(labels, kM.labels_)
    print("Homogenity: " , hg )
    print("Completeness: " , cp )
    print("V-measure: " , v )

#-------------------------------------------------------------------------------
# Main program.
#-------------------------------------------------------------------------------

# Get all the data as a defined struct
d_mnist, d_har, d_pulsar = read_data()

# Print Shapes...
#
# for har,
print(d_har.X_train.shape)
print(d_har.Y_train.shape)
# for pulsar,
print(d_pulsar.X_train.shape)
print(d_pulsar.Y_train.shape)
# for mnist.
print(d_mnist.X_train.shape)
print(d_mnist.Y_train.shape)

print("Pulsar : ")
cluster_data(d_pulsar.X_train,d_pulsar.Y_train,8)

print("Har : ")
cluster_data(d_har.X_train,d_har.Y_train, 6)

print("Mnist : ")
cluster_data(d_mnist.X_train,d_mnist.Y_train, 10)

# l = list(data2)
# d, axes = plt.subplots(2, 4)
# d.set_size_inches(20.5, 10.5)

# for i in range(0,2):
#     for j in range(0,4):
#         sns.boxplot(data=data2,
#                       x='8',
#                       y=l[(i*2)+j],ax=axes[i][j])
# plt.show()

#************ FOR TRAINING **************#
# df_dummies = pd.get_dummies(data)
# print(df_dummies.head())
# df_dummies.drop(['encounter_id', 'patient_nbr'], axis=1, inplace = True)
# print(df_dummies.head())
# print()
# np.savetxt("dummy_out.txt", df_dummies.columns.to_numpy(),fmt="%s")



#-------------------------------------------------------------------------------
#                               End of aec.py
#-------------------------------------------------------------------------------
