'''
importing libraries
'''
import numpy as np
import pandas as pd
from math import inf as infinity


'''
loading training data
'''
dataset = pd.read_csv('train.csv')
X_train = dataset.iloc[:,0:4].values
y_train = dataset.iloc[:,4].values


'''
loading test data
'''
dataset = pd.read_csv('test.csv')
X_test = dataset.iloc[:,0:4].values
y_test = dataset.iloc[:,4].values


'''
mapping the classes to an integer to index them easily
'''
class_map = {
    'Iris-setosa' : 0,
    'Iris-virginica' : 1,
    'Iris-versicolor' : 2
}

reverse_map = {
	0 : 'Iris-setosa',
    1 : 'Iris-virginica',
    2 : 'Iris-versicolor'
}


'''
grouping the features by the classes they belong to
after this we can easily get the class conditionals
'''
def groupByClass(data_features, data_classes):
	groupedByClass = []
	for _ in range(3):
		groupedByClass.append([])

	for _ in range(len(data_features)):
		groupedByClass[class_map[data_classes[_]]].append(data_features[_])

	return groupedByClass



'''
we create a gaussian distribution with the given mean and sigma
this step is to get the class conditionals
'''
def fitGaussian(data):
	mu = np.mean(data)
	sigma = np.std(data)
	return [mu,sigma]



'''
getting the pdf of a given value from the respective distribution
'''
def pdf(mu,sigma,x):
	return (1/(np.sqrt(2*np.pi)*sigma))*np.exp((-1/2)*(((x-mu)/sigma)**2))




'''
getting the probability for a given class
'''
def predict(test_element,distribution):
	ans = 1
	for attribute in range(4):
		mu, sigma = distribution[attribute][0],distribution[attribute][1]
		ans *= pdf(mu,sigma,test_element[attribute])
	return ans



def main():
	'''
	driver code starts here
	''' 
	groupedByClass = groupByClass(X_train,y_train) #group the data to get class conditionals

	classConditionals = [] #class-conditional matrix, contains mean and sigma for every possible likelihood
	for i in range(3):
	    temp = []
	    for j in range(4):
	        temp.append(0)
	    classConditionals.append(temp)

	for theClass in range(3): #populating the matrix by class conditional info
	    for attribute in range(4):
	        temp = []
	        for _ in groupedByClass[theClass]:
	            temp.append(_[attribute])
	        classConditionals[theClass][attribute] = fitGaussian(temp)

	y_pred = [] #predicted class list
	for _ in X_test:
	    ans = -infinity
	    answerClass =  0
	    for theClass in range(3):
	        prediction = predict(_,classConditionals[theClass])
	        if prediction > ans: #choosing the class for which the posteriori is maximum
	            ans = prediction
	            answerClass = theClass
	    y_pred.append(reverse_map[answerClass])

	cm = np.zeros(9).reshape((3,3)) #confusion matrix
	trueCount = 0
	for _ in range(len(y_pred)): #populating the confusion matrix
	    cm[class_map[y_pred[_]]][class_map[y_test[_]]] += 1
	    if(class_map[y_test[_]] == class_map[y_pred[_]]):
	        trueCount += 1;

	print("Number of correct predictions : ", trueCount, "\n")

	print("confusion matrix:\n================")
	print(cm,"\n")
	print("efficiency = {}".format(trueCount/len(y_test))) #effciency


if __name__ == '__main__':
	main()