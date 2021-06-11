import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

#PERFORMING EDA


#defining dataset
iris=datasets.load_iris()

#converting it into pandas dataframe for ease
df=pd.DataFrame(data=np.c_[iris['data'],iris['target']],columns=iris['feature_names']+['target'])

#splitting data in x and y (x=features and y=response)
x=df.iloc[:,:-1]
y=df.iloc[:,-1]

#splitting data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True,random_state=0)

#converting x,y train and test into numpy arrays
x_train=np.asarray(x_train)
x_test=np.asarray(x_test)
y_train=np.asarray(y_train)
y_test=np.asarray(y_test)

#normalizing our features in x (y is already target, it doesn't need normalization)
scaler=Normalizer().fit(x_train)
nor_x_train=scaler.transform(x_train)
nor_x_test=scaler.transform(x_test)
print(nor_x_test)

#visualization using seaborn after normalization
df_2=pd.DataFrame(data=np.c_[nor_x_train,y_train],columns=iris['feature_names']+['target'])
dict={0.0:'Setosa',1.0:'Versicolor',2.0:'Virginica'}
after=sns.pairplot(df_2.replace({'target':dict}),hue='target')
after.fig.suptitle("pair plots",y=1.08)
plt.show()

#ALGORITHM IMPLEMENTATION BEGINS

#1. Calculating Euclidean distance

def dist(x_train,x_test_point):
    d=[]
    for row in range (len(x_train)):
        current_train_point=x_train[row]
        current_distance=0
        for col in range (len(current_train_point)):
            current_distance+=(current_train_point[col] - x_test_point[col])**2
        current_distance=np.sqrt(current_distance)
        d.append(current_distance)
    d=pd.DataFrame(data=d,columns=['dist'])
    return d

#2. Finding the k nearest neighbours
def near(dist_pt,k):
    df_nearest=dist_pt.sort_values(by=['dist'],axis=0)
    df_nearest=df_nearest[:k]
    return df_nearest

#3. Classify point based on majority vote

def voting(df_nearest,y_train):
    counter_vote=Counter(y_train[df_nearest.index])
    y_prediction=counter_vote.most_common()[0][0]
    return y_prediction

def KNN_full(x_train,y_train,x_test,k):
    y_prediction=[]
    for x_test_point in x_test:
        dist_pt=dist(x_train,x_test_point)
        df_nearest_pt=near(dist_pt,k)
        y_prediction_pt=voting(df_nearest_pt,y_train)
        y_prediction.append(y_prediction_pt)
        return y_prediction
k=3
y_prediction_scratch=KNN_full(nor_x_train,y_train,nor_x_test,k)
print(y_prediction_scratch)














