from sklearn import tree

features=[[35,1],[47,1],[90,1],[48,1],[90,0],[35,1],[92,1],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]
labels=[1,2,2,1,2,1,2,1,1,1,2,1,2,1,2]

#load the dataset
#Rough 1
#Smooth 0

#Tennis 1
#Cricket 2

obj=tree.DecisionTreeClassifier()

obj=obj.fit(features,labels)

print(obj.predict([[97,0],[35,1]]))