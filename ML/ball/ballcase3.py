from sklearn import tree

def Ballpredictor(weight,surface):
    features=[[35,1],[47,1],[90,1],[48,1],[90,0],[35,1],[92,1],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]
    labels=[1,2,2,1,2,1,2,1,1,1,2,1,2,1,2]

    #load the dataset
    #Rough 1
    #Smooth 0

    #Tennis 1
    #Cricket 2

    obj=tree.DecisionTreeClassifier()

    obj=obj.fit(features,labels)

    ret=(obj.predict([[weight,surface]]))
    if ret==1:
        print("Your object looks like a Tennis Ball")
    else:
        print("Your object looks like a Cricket ball")    

def main():
    print("____________ball predictor case Study_____________")
    print("Please enter the weight of your object in grams")
    weight=int(input())

    print("please enter the type of surface of your object(Rough/smooth) ")
    surface=input()

    if surface.lower()=="rough":
        surface=1
    elif surface.lower()=="smooth":
        surface=0
    else:
        print("Invalid type of surface")
        exit() 

    Ballpredictor(weight,surface)



if __name__=="__main__":
    main()
