#_______________________________________________________________________________
# CE888 Project |        aec.py       | Ogulcan Ozer. | Feb. 2019 | UNFINISHED.
#_______________________________________________________________________________

import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
##************************
import tensorflow as tf
from keras import utils as ku
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.utils import plot_model
from sklearn import cluster, datasets, metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,train_test_split)
from sklearn.preprocessing import StandardScaler
from tensorflow.examples.tutorials.mnist import input_data
##*************************
from ae import make_ae
from ae_config import ae_param as PAR
from data import data_struct as DS

##************************
f = open('out.txt','a')
runtime = 'RUN: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(runtime, file = f)
#np.random.seed(42)
path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))
results = pd.DataFrame(columns=['Clustering Method','Homogenity','Completeness','V-Measure','Accuracy']) 
#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def read_data():

    #Read and seperate the mnist dataset.
    (X_train, Y_train), (x_test, y_test) = mnist.load_data()
    
    #Flatten the data, MLP doesn't use the 2D structure of the data. 784 = 28*28
    X_train = X_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    
    #Make the value floats in [0;1] instead of int in [0;255]
    tr = X_train.astype('float32')
    te = x_test.astype('float32')
    tr = tr / 255
    te = te / 255
    trl = np.asarray(Y_train, dtype=np.int32)
    tel = np.asarray(y_test, dtype=np.int32)
    print(tr[1,:])
    mnist_data = DS(tr,trl,te,tel,'mnist')
    print(tr)
    print(trl)
    
    #Read and seperate the HAR dataset.
    tr1 = pd.read_csv(os.path.join(path,'UCI HAR Dataset','train',
    'X_train.txt'),delim_whitespace=True,error_bad_lines=False, warn_bad_lines=False,
    header=None)
    tr1 = tr1.to_numpy(dtype='float32', copy=True)
    trl1 = pd.read_csv(os.path.join(path,'UCI HAR Dataset','train',
    'y_train.txt'), error_bad_lines=False, warn_bad_lines=False,header=None)
    te1 = pd.read_csv(os.path.join(path,'UCI HAR Dataset','test',
    'X_test.txt'),delim_whitespace=True,error_bad_lines=False, warn_bad_lines=False,
    header=None)
    te1 = te1.to_numpy(dtype='float32', copy=True)
    tel1 = pd.read_csv(os.path.join(path,'UCI HAR Dataset','test',
    'y_test.txt'),error_bad_lines=False, warn_bad_lines=False, header=None)
    har_data = DS(tr1,trl1.values.T[0],te1,tel1.values.T[0],'har')

    #Read and seperate the pulsar dataset.
    alldata = pd.read_csv(os.path.join(path,'HTRU2','HTRU_2.csv'),
    sep=',',error_bad_lines=False, warn_bad_lines=False,header=None)
    labels = pd.DataFrame(alldata[alldata.columns[8]])
    alldata.drop(alldata.columns[8], axis = 1, inplace = True)
    tr2, te2, trl2, tel2 = train_test_split(alldata, labels, test_size=0.20,
    random_state=42)
    tr2 = tr2.to_numpy(dtype='float32', copy=True)
    te2 = te2.to_numpy(dtype='float32', copy=True)
  
    pulsar_data = DS(tr2,trl2.values.T[0],te2,tel2.values.T[0],'pulsar')

    return mnist_data, har_data, pulsar_data

#Function for Kmeans clustering
def cluster_data(_data, cluster_size,results):

    p =  'K_Means_Data_Cluster'
    im = _data.name + p
    data = _data.X_train
    labels = _data.Y_train
    # kmeans clustering 
    kM=cluster.KMeans(init='k-means++', n_clusters = cluster_size, n_init=10, n_jobs=4)
    y_pred_kmeans = kM.fit_predict(data)
    rs = scores(_data.name,labels,y_pred_kmeans,results)
    sns.scatterplot(x=data[:,0], y=data[:,1],
    hue=kM.labels_, legend="full")
    sns.scatterplot(x=kM.cluster_centers_[:,0], y=kM.cluster_centers_[:,1],
    color='black')
    plt.title(im, fontsize=15)
    plt.grid(True)
    plt.savefig('%s.png' % im , dpi=300)
    plt.show()

    return kM, rs

#Function for getting and plotting the pca variance ratio 
def pca_comp(_data):

    p =  'pca'
    im = _data.name + p
    data = _data.X_train
    pca = PCA()
    reduced_train = pca.fit_transform(data)
    plt.bar(range(0, data.shape[1]), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.step(range(0, data.shape[1]), np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.savefig('%s.png' % im , dpi=300)
    plt.show()
    return reduced_train

#Function for applying n_component pca to a given data
def pca_linear(_data, n_components):
    train = _data.X_train
    test = _data.x_test
    name = 'pca_' + _data.name
    pca = PCA(n_components=n_components)
    reduced_train = pca.fit_transform(train)
    reduced_test = pca.transform(test)
    pca_data = DS(reduced_train,_data.Y_train,reduced_test,_data.y_test,name)
    return pca_data

def ae_build(_data,encoding,optimizer,loss,_type):
    if(_data.name == 'mnist'):
        train = _data.X_train
        test = _data.x_test
    else:
        train = _data.X_train_std
        test = _data.x_test_std
    if(_type == 'linear'):
        name = 'aelinear_' + _data.name
    else:
        name = 'aenon-linear_' + _data.name
    ae = make_ae(train.shape[1],[encoding],encoding,[train.shape[1]],[_type],[PAR.af_adec])
    ae.autoencoder.compile(optimizer=optimizer, loss=loss)
    hist = ae.autoencoder.fit(train, train,
                    epochs=PAR.ae_epoch,
                    batch_size=PAR.ae_batch,
                    shuffle=True,
                    validation_data=(test, test))
    plot_ae( name, hist)
    # encoded_imgs = ae.encoder.predict(x_test)
    # decoded_imgs = ae.decoder.predict(encoded_imgs)
    reduced_train = ae.encoder.predict(train)
    reduced_test = ae.encoder.predict(test)
    ae_data = DS(reduced_train,_data.Y_train,reduced_test,_data.y_test,name)
    return ae_data, ae

def ae_softmax(_data,encoding,optimizer,loss):
    if(_data.name == 'mnist'):
        train = _data.X_train
        test = _data.x_test
    else:
        train = _data.X_train_std
        test = _data.x_test_std
    name = 'aesoftmax_' + _data.name
    ae = make_ae(train.shape[1],[encoding],encoding,[train.shape[1]],['softmax'],['relu'])
    ae.autoencoder.compile(optimizer=optimizer, loss=loss)
    hist = ae.autoencoder.fit(train, train,
                    epochs=PAR.ae_epoch,
                    batch_size=PAR.ae_batch,
                    shuffle=True,
                    validation_data= (test, test))
    plot_ae( name, hist)
    reduced_train = ae.encoder.predict(train)
    train_p = np.argmax(reduced_train, axis =1)
    reduced_test = ae.encoder.predict(test)
    test_p = np.argmax(reduced_test, axis=1)
    return train_p, test_p, ae

#Function for checking the response of the softmax layer for each class
def softmax_pred(_data, ae, results):
    if(_data.name == 'mnist'):
        dic = {label: _data.x_test[_data.y_test==label] for label in np.unique(_data.y_test)}
    else:
        dic = {label: _data.x_test_std[_data.y_test==label] for label in np.unique(_data.y_test)}
    num_classes = len(np.unique(_data.y_test))
    for i in range(num_classes):
        resf=ae.encoder.predict(dic[i])
        res = np.argmax(resf, axis =1)
        for j in range(num_classes):
            name = 'Class' + str(i) + 'vs' + 'Node' + str(j)
            scores(name, np.full(len(res), j, dtype='float32'), res , results)



def scores(name ,y_true, y_pred,results):

    hg, cp, v = metrics.homogeneity_completeness_v_measure(y_true, y_pred)
    a = metrics.accuracy_score(y_true, y_pred)
    print(name)
    print(name , file = f)
    print("Homogenity: " , hg )
    print("Completeness: " , cp )
    print("V-measure: " , v )
    print("Homogenity: " , hg, file = f)
    print("Completeness: " , cp, file = f)
    print("V-measure: ", file = f)
    print("Accuracy: " , a )
    print("Accuracy: " , a, file = f)
    results = results.append({'Clustering Method': name ,'Homogenity':hg,'Completeness':cp,'V-Measure':v,'Accuracy':a}, ignore_index=True)
    return results

def plot_ae(name,ae):

    # Plot training & validation loss values
    plt.plot(ae.history['loss'])
    plt.plot(ae.history['val_loss'])
    plt.title(name)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('%s.png' % name , dpi=300)
    plt.show()

#-------------------------------------------------------------------------------
# Main program.
#-------------------------------------------------------------------------------

# Get all the data as a defined struct
d_mnist, d_har, d_pulsar = read_data()
d_har.Y_train[:] = [x - 1 for x in d_har.Y_train]
d_har.y_test[:] = [x - 1 for x in d_har.y_test]
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

# print("Pulsar : ")
# print("Pulsar : ", file = f)
# _,results = cluster_data(d_pulsar, 2,results)

# print("Har : ")
# print("Har : ", file = f)
# _,results = cluster_data(d_har, 6,results)

# print("Mnist : ")
# print("Mnist : ", file = f)
# _,results = cluster_data(d_mnist, 10,results)


# #*** Clustering with linear PCA ***#
# #check component variances#
# pca_comp(d_pulsar)
# pca_comp(d_har)
# pca_comp(d_mnist)

# p_pulsar = pca_linear(d_pulsar, 4)
# p_har = pca_linear(d_har, 200)
# p_mnist = pca_linear(d_mnist, 400)

# print("Pulsar pca : ")
# print("Pulsar pca : ", file = f)
# _,results =cluster_data(p_pulsar, 2,results)

# print("Har pca : ")
# print("Har pca: ", file = f)
# _,results = cluster_data(p_har, 6,results)

# print("Mnist pca: ")
# print("Mnist pca: ", file = f)
# _,results =cluster_data(p_mnist, 10,results)

#*** Clustering using AE with linear activation functions (~PCA) and kmeans***#


# ae_pulsar, ae_p = ae_build(d_pulsar, 4, PAR.ae_opt, PAR.ae_loss,'linear')
# ae_har, ae_h = ae_build(d_har, 200, PAR.ae_opt, PAR.ae_loss,'linear')
# ae_mnist, ae_m = ae_build(d_mnist, 400, PAR.ae_opt, PAR.ae_loss,'linear')



# print("Pulsar ae : ")
# print("Pulsar ae : ", file = f)
# _,results =cluster_data(ae_pulsar, 2,results)

# print("Har ae : ")
# print("Har ae: ", file = f)
# _,results =cluster_data(ae_har, 6,results)

# print("Mnist ae: ")
# print("Mnist ae: ", file = f)
# _,results =cluster_data(ae_mnist, 10,results)

#*** Clustering using AE with non-linear activation functions and kmeans***#
nlae_pulsar, nlae_p = ae_build(d_pulsar, 4, PAR.ae_opt, PAR.ae_loss,'sigmoid')
nlae_har, nlae_h = ae_build(d_har, 200, PAR.ae_opt, PAR.ae_loss,'sigmoid')
nlae_mnist, nlae_m = ae_build(d_mnist, 400, PAR.ae_opt, PAR.ae_loss,'sigmoid')

print("Pulsar non-linear ae : ")
print("Pulsar non-linear ae : ", file = f)
_,results =cluster_data(nlae_pulsar, 2,results)

print("Har non-linear ae : ")
print("Har non-linear ae: ", file = f)
_,results =cluster_data(nlae_har, 6,results)

print("Mnist non-linear ae: ")
print("Mnist non-linear ae: ", file = f)
_,results =cluster_data(nlae_mnist, 10,results)

# #*** Clustering using AE with softmax feature layer***#
# p_train_pred, p_test_pred, sf_p = ae_softmax(d_pulsar, 2, PAR.ae_opt, PAR.ae_loss)
# h_train_pred, h_test_pred, sf_h = ae_softmax(d_har, 6, PAR.ae_opt, PAR.ae_loss)
# m_train_pred, m_test_pred, sf_m = ae_softmax(d_mnist, 10, PAR.ae_opt, PAR.ae_loss)

# softmax_pred(d_pulsar,sf_p,results)
# softmax_pred(d_har,sf_h,results)
# softmax_pred(d_mnist,sf_m,results)


# print("Pulsar ae_softmax test: ")
# print("Pulsar ae_softmax test: ", file = f)
# results =scores(d_pulsar.name, p_test_pred,d_pulsar.y_test,results)

# print("Har ae_softmax test: ")
# print("Har ae_softmax test: ", file = f)
# results =scores(d_har.name, h_test_pred,d_har.y_test,results)

# print("Mnist ae_softmax test: ")
# print("Mnist ae_softmax test: ", file = f)
# results =scores(d_mnist.name, m_test_pred,d_mnist.y_test,results)


print(results.to_latex(index=False))
print(results.to_latex(index=False), file = f)




#-------------------------------------------------------------------------------
#                               End of aec.py
#-------------------------------------------------------------------------------
