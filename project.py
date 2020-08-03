import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from mpi4py import MPI
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
start_time = time.time()
if(rank == 0):
    df = pd.read_csv('diabetes.csv')
    df.head()
    X = df.drop(['Outcome'],axis = 1)
    y = df['Outcome']
    
    SCALE = StandardScaler()
    SCALE.fit(X)
    X_scale = SCALE.transform(X)
    X_scale = pd.DataFrame(X_scale)
    x_train,x_test,y_train,y_test = train_test_split(X_scale,y,test_size = 0.2,random_state = 0)
   
    param_grid = [{'C' : [0.001,0.01,0.1,1,10,100], 'kernel' : ['linear', 'rbf']}]
    svc = SVC()
    clf = GridSearchCV(svc, param_grid, cv = 8)
    clf.fit(x_train, y_train)


    # predict the target on the train dataset
    predict_train = clf.predict(x_train)
    #print('Target on train data',predict_train) 

    # Accuracy Score on train dataset
    accuracy_train = accuracy_score(y_train,predict_train)
    print('accuracy_score on train dataset : ', accuracy_train)
    print('')

    # predict the target on the test dataset
    predict_test = clf.predict(x_test)
    #print('Target on test data',predict_test) 

    # Accuracy Score on test dataset
    accuracy_test = accuracy_score(y_test,predict_test)
    print('accuracy_score on test dataset : ', accuracy_test)

    print('')
    print('Best set of parameters are')
    print(clf.best_params_)

    print('The time taken for this program to run is: ',time.time()-start_time)
    
    #svc = SVC(kernel = 'linear')
    #svc.fit(x_train, y_train)
    #predict_train = svc.predict(x_train)
    #yhat = svc.predict(x_test)

    # Accuracy Score on train dataset
    #accuracy_train = accuracy_score(y_train,predict_train)
    #print('accuracy_score on train dataset : ', accuracy_train)

    # Accuracy Score on test dataset
    #accuracy_test = accuracy_score(y_test,yhat)
    #print('accuracy_score on test dataset : ', accuracy_test)