from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import pandas as pd

import numpy as np
import keras as k
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from numpy.random import seed
seed(1)


global model,balance_data,balance_data1,balance_data2,test_x

def index(request):
    if request.method == 'GET':
       return render(request, 'Home.html', {})

def User(request):
    if request.method == 'GET':
       return render(request, 'UserPannel.html', {})


def Admin(request):
    if request.method == 'GET':
       return render(request, 'AdminLogin.html', {})

def AdminLogin(request):
    if request.method == 'POST':
      username = request.POST.get('username', False)
      password = request.POST.get('password', False)
      if username == 'admin' and password == 'admin':
       context= {'data':'welcome '+username}
       return render(request, 'AdminPannel.html', context)
      else:
       context= {'data':'login failed\nCheck username/Password '}
       return render(request, 'AdminLogin.html', context)



def importdata(): 
    balance_data = pd.read_csv('C:/FakeProfile/Profile/dataset/dataset.txt')
    balance_data = balance_data.abs()
    rows = balance_data.shape[0]  # gives number of row count
    cols = balance_data.shape[1]  # gives number of col count
    return balance_data 
    
##    ##adding isFake column
##    isNotFake = np.zeros(299)
##    isFake = np.ones(299)
##
##    print("assigning of 0 & 1's completed")
##    
##    balance_data2["isFake"] = isFake
##    balance_data1["isFake"] = isNotFake
##    
##    print("Completed adding is fake or not column to make predictions for it")
##    ##Combining different datasets into one
##    balance_data = pd.concat([balance_data1 , balance_data2] , ignore_index = True)
##    balance_data.columns = balance_data.columns.str.strip()
##    balance_data = balance_data.sample(frac=1).reset_index(drop=True)
##    balance_data.describe()
##    print(balance_data.head())
##    
##    return balance_data

def splitdataset(balance_data):
    X = balance_data.values[:, 0:8] 
    y_= balance_data.values[:, 8]
    y_ = y_.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(y_)
    print(Y)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

    return train_x, test_x, train_y, test_y


##    X = balance_data.values[:, 0:8] 
##    y_= balance_data.values[:, 8]
##    y_ = y_.reshape(-1, 1)
##    encoder = OneHotEncoder(sparse=False)
##    Y = encoder.fit_transform(y_)
##    print(Y)
    
##    
##    y = balance_data.isFake
##    balance_data.drop(["isFake"], axis=1, inplace=True)
##    x = balance_data
##    
####    profile = ProfileReport(X, title="Pandas Profiling Report")
####    profile
##    
##    y.reset_index(drop=True, inplace=True)
##    print("Y shape : ",y.shape)
##    print("x shape : ",x.shape)
##    print(x.head())
##
##   
##    train_x, test_x, train_y, test_y = train_test_split(x , y , train_size=0.8, test_size=0.2, random_state=0)
##    return train_x, test_x, train_y, test_y
##
def UserCheck(request):
    if request.method == 'POST':
      data = request.POST.get('t1', False)
      input = 'Account_Age,Gender,User_Age,Link_Desc,Status_Count,Friend_Count,Location,Location_IP\n';
      input+=data+"\n"
      f = open("C:/FakeProfile/Profile/dataset/test.txt", "w")
      f.write(input)
      f.close()
      test = pd.read_csv('C:/FakeProfile/Profile/dataset/test.txt')
      test = test.values[:, 0:8] 
      predict = model.predict_classes(test)
      print(predict[0])
      msg = ''
      if str(predict[0]) == '1':
         msg = "Given Account Details Predicted As Genuine"
      if str(predict[0]) == '0':
         msg = "Given Account Details Predicted As Phishing account"
      context= {'data':msg}
      return render(request, 'UserPannel.html', context)

##def Att_selection():
##        ##Feature selection 
##    x = x[["Account_Age","Gender","User_Age","Link_Desc","Status_Count","Friend_Count","Location","Location_IP","Status"]]
##    x = x.replace(np.nan , 0)
##
##    return x

def GenerateModel(request):
    global model
    data = importdata()
    train_x, test_x, train_y, test_y = splitdataset(data)
    model = Sequential()
    model.add(Dense(200, input_shape=(8,), activation='relu', name='fc1'))
    model.add(Dense(200, activation='relu', name='fc2'))
    model.add(Dense(2, activation='sigmoid', name='output'))
    optimizer = Adam(lr=0.001)
    
    
    print('Artifical Neural Network Model Summary: ')
    print(model.summary())
    
    ## Compile_Model
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print("Compiled model")
    ## model_training
##    train_x = np.array(train_x)
##    train_y = np.array(train_y)
##    print(train_x.shape)
##    print(train_x.shape)
    
    model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200)
    results = model.evaluate(test_x, test_y)
    ann_acc = results[1] * 100

    ##model_testing
    score = model.evaluate(test_x, test_y, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    ##Graphs
##    plt.plot(history.history['accuracy'])
##    plt.plot(history.history['val_accuracy'])
##    plt.title('Model accuracy')
##    plt.ylabel('Accuracy')
##    plt.xlabel('Epoch')
##    axes = plt.gca()
##    axes.set_xlim([0,14])
##    axes.set_ylim([0.85,1])
##    axes.grid(True, which='both')
##    axes.axhline(y=0.85, color='k')
##    axes.axvline(x=0, color='k')
##    axes.axvline(x=14, color='k')
##    axes.axhline(y=1, color='k')
##    plt.legend(['Train','Val'], loc='lower right')
##    plt.show()
    
    
    ##context= {'data':'ANN Accuracy : '+str(ann_acc)}
    context= {'data':'ANN model generated successfully.... '}
    return render(request, 'AdminPannel.html', context)

 

def ViewTrain(request):
    if request.method == 'GET':
       strdata = '<table border=1 align=center width=100%><tr><th><font size=4 color=white>Account Age</th><th><font size=4 color=white>Gender</th><th><font size=4 color=white>User Age</th><th><font size=4 color=white>Link Description</th> <th><font size=4 color=white>Status Count</th><th><font size=4 color=white>Friend Count</th><th><font size=4 color=white>Location</th><th><font size=4 color=white>Location IP</th><th><font size=4 color=white>Profile Status</th></tr><tr>'
       data = pd.read_csv('C:/FakeProfile/Profile/dataset/dataset.csv')
       rows = data.shape[0]  # gives number of row count
       cols = data.shape[1]  # gives number of col count
       for i in range(rows):
          for j in range(cols):
             strdata+='<td><font size=3 color=white>'+str(data.iloc[i,j])+'</font></td>'
          strdata+='</tr><tr>'
       context= {'data':strdata}
       return render(request, 'ViewDataAdmin.html', context)
