#_______________________________________________________________________________
# CE888 Project |        aec.py       | Ogulcan Ozer. | Feb. 2019 | UNFINISHED.
#_______________________________________________________________________________

##************************
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
##************************

path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))
#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
n
#-------------------------------------------------------------------------------
# Main program.
#-------------------------------------------------------------------------------

#Read and seperate the diabetes dataset.
data = pd.read_csv(os.path.join(path,'dataset_diabetes\diabetic_data.csv'),
sep=',',error_bad_lines=False, warn_bad_lines=False)
target = pd.DataFrame(data['readmitted'])
data.drop(['readmitted'], axis = 1, inplace = True)
print(data.head())
print(data.describe())

print('-----------------------------------------------------------------------')

#Read and seperate the HAR dataset.
data1 = pd.read_csv(os.path.join(path,'UCI HAR Dataset','train',
'X_train.txt'),delim_whitespace=True,error_bad_lines=False, warn_bad_lines=False,
header=None)
target1 = pd.read_csv(os.path.join(path,'UCI HAR Dataset','train',
'y_train.txt'),sep=',',error_bad_lines=False, warn_bad_lines=False)
data1['labels'] = target1 
print(data1.head())
print(data1.describe())


print('-----------------------------------------------------------------------')

#Read and seperate the pulsar dataset.
data2 = pd.read_csv(os.path.join(path,'HTRU2\HTRU_2.csv'),
sep=',',error_bad_lines=False, warn_bad_lines=False,header=None)
target2 = pd.DataFrame(data2[data2.columns[8]])
#data2.drop(data2.columns[8], axis = 1, inplace = True)
print(data2.head())
print(data2.describe())
data2.rename(columns=lambda x: str(x),inplace = True)

l = list(data2)
d, axes = plt.subplots(2, 4)
d.set_size_inches(20.5, 10.5)

for i in range(0,2):
    for j in range(0,4):
        sns.boxplot(data=data2,
                      x='8',
                      y=l[(i*2)+j],ax=axes[i][j])
plt.show()

#************ FOR TRAINING **************#
# df_dummies = pd.get_dummies(data)
# print(df_dummies.head())
# df_dummies.drop(['encounter_id', 'patient_nbr'], axis=1, inplace = True)
# print(df_dummies.head())
# print()
# np.savetxt("dummy_out.txt", df_dummies.columns.to_numpy(),fmt="%s")
#************* USE TO SAVE AND DISPLAY THE MODEL *************#
# writer = tf.summary.FileWriter('.')
# writer.add_graph(tf.get_default_graph())
# writer.flush()

#-------------------------------------------------------------------------------
#                               End of aec.py
#-------------------------------------------------------------------------------
