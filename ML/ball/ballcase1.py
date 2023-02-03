from sklearn import tree

features=[[35,"Rough"],[47,"Rough"],[90,"Smooth"],[48,"Rough"],[90,"Smooth"],[35,"Rough"],[92,"Smooth"],[35,"Rough"],[35,"Rough"],[35,"Rough"],[96,"Smooth"],[43,"Rough"],[110,"Smooth"],[35,"Rough"],[95,"Smooth"]]
labels=["Tennis","Tennis","Cricket","Tennis","Cricket","Tennis","Cricket","Tennis","Tennis","Tennis","Cricket","Tennis","Cricket","Tennis","Cricket"]



obj=tree.DecisionTreeClassifier()

obj=obj.fit(features,labels)

print(obj.predict([[97,"Smooth"]]))