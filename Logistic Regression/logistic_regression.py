import pandas as pd
import numpy as np


def sigmoid(z):
    """calculates the sigmoid value"""
    func = 1/(1+np.exp(-z))
    return func


def delta_loss(a,b,x):
    """ calculates the gradient factor"""
    return np.dot(x,(a-b).T)
    
def gradient_cons1(y,h_theta,aug_feat_space,weight_vector):
    """updates the parameter until gradient factor becomes less than 0.01"""
    loss = delta_loss(y,h_theta,aug_feat_space)
    l_r = 0.02
    while abs(np.mean(loss)) > 0.01:
        weight_vector = weight_vector+l_r*loss
        z = np.dot(weight_vector.T,aug_feat_space)
        h_theta = sigmoid(z)
        loss = delta_loss(y,h_theta,aug_feat_space)
    return weight_vector

def gradient_cons2(y,h_theta,aug_feat_space,weight_vector):
    """updates weights for 80 epochs"""
    loss = delta_loss(y,h_theta,aug_feat_space)
    l_r = 0.02
    epoch = 80
    for i in range(epoch):
        weight_vector = weight_vector+l_r*loss
        z = np.dot(weight_vector.T,aug_feat_space)
        h_theta = sigmoid(z)
        loss = delta_loss(y,h_theta,aug_feat_space)
    return weight_vector
def accuracy(y,y_das):
    """calculates accuracy of the model"""
    tp = 0
    fp = 0
    for i in range(len(y_test)):
        if y_test[i]==y_das[0][i]:
            tp+=1
        else:
            fp+=1
    return tp/(tp+fp)*100

def initilaize_parametrs(aug_feat_space):
    """Initialize the parameters with zero value"""
    weight_vector = np.zeros((aug_feat_space.shape[0],1))
    z = np.dot(weight_vector.T,aug_feat_space)
    a = sigmoid(z)

    return weight_vector,z,a

      



if __name__ == "__main__":

    train_df = pd.read_excel('train_2.xlsx')
    test_df = pd.read_csv('test_2.csv')

    train_df.iloc[:,4] = train_df.iloc[:,4].map({'Iris-virginica':1,'Iris-versicolor':0})
    test_df.iloc[:,4] = test_df.iloc[:,4].map({'Iris-virginica':1,'Iris-versicolor':0})

    

    aug_feat_space = np.insert(train_df.iloc[:,:4].values,0,1,axis = 1).T


    x = train_df.iloc[:,:4]
    y = (train_df.iloc[:,4].values).reshape(1,x.shape[0])

    x_test = test_df.iloc[:,:4].values
    x_test_aug = np.insert(x_test,0,1,axis = 1).T

    """
    if  gradient value is less than 0.01
    """
    weight_vector,z,h_theta = initilaize_parametrs(aug_feat_space)
    weight = gradient_cons1(y,h_theta,aug_feat_space,weight_vector)
    z_pred = np.dot(weight.T,x_test_aug)
    y_hat = sigmoid(z_pred)
    y_pred_1 = (y_hat>0.5).astype(int)
    y_test =test_df.iloc[:,4].values

    print("Test set accuracy is for first stopping criteria is  {:.2f}".format(accuracy(y_test,y_pred_1)))

    """
    if  epoch is equal to 80
    """
    weight_vector,z,h_theta = initilaize_parametrs(aug_feat_space)
    weight = gradient_cons2(y,h_theta,aug_feat_space,weight_vector)
    z_pred = np.dot(weight.T,x_test_aug)
    y_hat = sigmoid(z_pred)
    y_pred_2 = (y_hat>0.5).astype(int)
    y_test =test_df.iloc[:,4].values

    print("Test set accuracy is for second stopping criteria is  {:.2f}".format(accuracy(y_test,y_pred_2)))
