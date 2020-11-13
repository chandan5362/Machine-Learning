'''
importing the libraries
'''
import numpy as np
import pandas as pd

'''
importing data
'''
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

'''
mapping the classes to an integer
to index them effciently
'''
train_df =train_df.replace({'Iris-virginica':1,'Iris-setosa':2,'Iris-versicolor':3})
test_df =test_df.replace({'Iris-virginica':1,'Iris-setosa':2,'Iris-versicolor':3})

'''
sorting the features by their class
'''
train_df.sort_values(by=['4'],inplace = True)



stats_tuple =[None]*4 #tuple of mu and sigma
cov_tuple = [None]*4 #covriance matrix
def populateTuples():
    for cat_ in train_df['4'].unique():
        series = train_df[train_df['4'] == cat_].iloc[:,:4]
        cov_tuple[cat_] = np.cov(series.T)
        stats_tuple[cat_] = [np.mean(series,axis =0),np.std(series,axis = 0)]
populateTuples()

#inversing the covrariance matrix
inv = np.linalg.inv(cov_tuple[1])
np.dot(cov_tuple[1],inv)


'''
function to calculate the log likelihood
'''
def find_log_liklihood_Bayesian(cov,mean,X):
    log_add = np.exp((-1/2)*(np.dot(np.dot((X.T-mean.T).T,np.linalg.inv(cov)),(X.T-mean.T))))
    cov_det = 1/(((2*np.pi)**2)*(np.linalg.det(cov))**0.5)
    return cov_det*log_add


'''
the matrix with posteriori values
'''
label = np.zeros((test_df.shape[0],4))
def populateLabel():
    for i in range(test_df.shape[0]):
        X = test_df.iloc[i,:4]
        for cl in range(1, len(stats_tuple)):
            mean_arr = np.array(stats_tuple[cl][0])
            label[i,cl] = np.prod(find_log_liklihood_Bayesian(cov_tuple[cl],mean_arr,X))

populateLabel()

'''
our prediction array
'''
bayesian_pred_arr  = np.argmax(label,axis = 1)
bayesian_pred_arr.shape[0]


cm = np.zeros(9).reshape((3,3)) #confusion matrix
tp = 0 #true predictions
for i in range(test_df.shape[0]):
    cm[bayesian_pred_arr[i]-1][test_df['4'].values[i]-1] += 1
    if bayesian_pred_arr[i] == test_df['4'].values[i]:
        tp+=1

print("Number of correct predictions : ", tp, "\n")
print("confusion matrix:\n================")
print(cm,"\n")
print("efficiency = {}".format(tp/bayesian_pred_arr.shape[0])) #effciency