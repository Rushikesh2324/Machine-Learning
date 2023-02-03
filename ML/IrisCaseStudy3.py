from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from Scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)




class MarvellousKNeighborsClassifier:

    def fit(self,trainingdata,trainingtarget ):
        self.TrainingData=trainingdata
        self.TrainingTarget=trainingtarget

    def Closest(self,row):
        minimumdistance=euc(row,self.TrainingData[0])
        minimumindex=0

        for i in range(1,len(self.TrainingData)):
            Distance=euc(row,self.TrainingData[i])
            if Distance<minimumdistance:
                minimumdistance=Distance
                minimumdistance=i

        return self.TrainingData[minimumindex]


    def predict(self,TestData):
        prediction=[]
        for value in TestData:
            result=self.Closest(value)
            prediction.append(result)

        return prediction



def MarvellousML():
    Dataset = load_iris()       # 1 Load the data

    Data = Dataset.data
    Target = Dataset.target

    Data_train, Data_test, Target_train, Target_test = train_test_split(Data, Target, test_size = 0.5)

    Classifier = KNeighborsClassifier()

    Classifier.fit(Data_train, Target_train)

    Predictions = Classifier.predict(Data_test)

    Accuracy = accuracy_score(Target_test, Predictions)

    return Accuracy

def main():
    Ret = MarvellousKNeighborsClassifier()

    print("Acuracy of Iris dataset with KNN is ",Ret * 100)

if __name__ == "__main__":
    main()