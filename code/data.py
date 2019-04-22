
#_______________________________________________________________________________
# CE888 Project |        data.py       | Ogulcan Ozer. | Feb. 2019 | UNFINISHED.
#_______________________________________________________________________________
from sklearn.preprocessing import StandardScaler
class data_struct:
    
    
    
    def __init__(self, X_train, Y_train, x_test, y_test,name):
        sc = StandardScaler()
        self.name = name
        self.X_train = X_train
        self.Y_train = Y_train
        self.x_test = x_test
        self.y_test = y_test
        self.X_train_std = sc.fit_transform(X_train)
        self.x_test_std = sc.transform(x_test)

#-------------------------------------------------------------------------------
# End of data.py  
#-------------------------------------------------------------------------------
