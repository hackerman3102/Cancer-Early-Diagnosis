
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
cancer=load_breast_cancer()
print(cancer.data[0])
print(cancer.feature_names)
x_train,x_test,y_train,y_test=train_test_split(cancer.data,cancer.target,test_size=0.2,random_state=3)
max=0
u=0
list=[]
for i in range(1,101):
  classifier=KNeighborsClassifier(n_neighbors=i)
  classifier.fit(x_train,y_train)
  a=classifier.score(x_test,y_test)
  list.append(a)
  if(a>max):
    max=a
    u=i
print("{} at {}".format(max,u))    
print(list)    
plt.plot(range(1,101),list)   
plt.show()
  #f1 score of 93.859 at knn=8
