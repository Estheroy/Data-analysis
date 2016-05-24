"""
Implement k-th nearest neighbors classifier
 
Input: train_X: a matrix for training set features 
       train_Y: a vector raining set class labels
       test_X: a vector for testing set features
       kVal: specify the k value in k-th nearest neighbors

Return: result: a vector for predicted class labels given features in toTest
 
Author: Xuanpei Ouyang
"""
import numpy as np
import random
import operator
from heapq import heappop
from heapq import heappush


"""
This function takes in test sample size, total sample size,
and random generator as parameters

Return: a list of index for the test sample
"""


def select_test_sample_index(test_size, total_size, random):
    test_sample_index = []

    i = j = 0
    tmp = test_size
    while i < tmp:
        prob = float(test_size)/float(total_size)
        rand = random.random()

        if rand <= prob:
            test_sample_index.append(j)
            test_size -= 1
            i += 1
        total_size -= 1
        j += 1

    return test_sample_index


"""
Given a n x 1 vector as original labels and convert original labels
into binary code labels by using one-hot encoding.

Return: mew_labels: the one-hot encoding of original labels
"""


def one_hot_encoding(labels):

    num_class = len(np.unique(labels))
    arr = np.zeros((len(labels), num_class), dtype=np.int)
    for i in range(len(labels)):
        arr[i][labels[i]] = 1

    return arr


"""
Given row training and testing data set,
Return training and testing data set with z scaling
"""


def z_transform(test_dataset, train_dataset):

    # calculate the mean and standard deviation of training sample
    mean = np.mean(test_dataset, axis=0)
    std = np.std(train_dataset, axis=0, ddof=1)

    for i in range(test_dataset.shape[1]-1):

        train_dataset[:, i] = (train_dataset[:, i] - mean[i]) / std[i]
        test_dataset[:, i] = (test_dataset[:, i] - mean[i]) / std[i]

    return train_dataset, test_dataset


"""
Given a data set
Return: n x 1 column which is the i th row of the dataset
"""


def get_column(data, i):
    return [row[i] for row in data]


"""
transform the category data to numerical data
"""


def categ_to_numerical(labels):

    new_categ = []
    for i in range(len(labels)):
        if labels[i] == 'F':
            new_categ.append(0)
        elif labels[i] == 'M':
            new_categ.append(1)
        else:
            new_categ.append(2)
    return new_categ


"""
Given two data points with n features

Return: the euclidian distance between these two data points
"""


def get_euclidian_distance(x1, x2):
    dist = np.sqrt(sum(np.square(x1 - x2))) 
    return dist


"""
Use KNN to predict testing data into different category
"""


def KNN(train_X, train_Y, test_X, kVal):

    print "\ntraining KNN"

    predict = np.zeros((test_X.shape[0]))
    for i in range(test_X.shape[0]):
        curr = test_X[i, :]
        pq = [] 
        for j in range(train_X.shape[0]):
            tmp_dist = get_euclidian_distance(curr, train_X[j,:])
            heappush(pq, (tmp_dist, j))
        indices = []
        for j in range(kVal):
            pair = heappop(pq)
            index = pair[1]
            indices.append(index)
        neighbor = {}

        for j in indices:
            if train_Y[j] in neighbor:
                count = neighbor[train_Y[j]]
                neighbor[train_Y[j]] = count+1
            else:
                neighbor[train_Y[j]] = 1
                
        sorted_neighbor = sorted(neighbor.items(), key=operator.itemgetter(1))
        pair = sorted_neighbor[len(sorted_neighbor)-1]
        pred = pair[0]
        predict[i] = pred

    return predict


"""
Calculate accuracy based on the result of KNN
"""


def accuracy(predict_Y, test_Y):

    correct = 0
    for i in range(test_Y.shape[0]):
        if predict_Y[i] == test_Y[i]:
            correct += 1

    acc = float(correct) / float(test_Y.shape[0])
    return acc


"""
This method will calculate confusion matrix
Only work for class labels that are already converted to [0, number of unqiue]
"""


def confusion_matrix(predict_Y, true_Y):

    labels = np.unique(np.hstack((true_Y, predict_Y)))

    dict={}
    for i in range(len(labels)):
        dict[labels[i]] = i

    num_of_labels = len(labels)
    mat = np.zeros((num_of_labels, num_of_labels), dtype=np.int)

    for i in range(predict_Y.shape[0]):
        true_class = int(true_Y[i])
        pred_class = int(predict_Y[i])

        true_index = dict[true_class]
        pred_index = dict[pred_class]
        tmp = mat[true_index][pred_index] + 1
        mat[true_index][pred_index] = tmp

    print "\nrow - expected label, col - predicted label"
    print "row labels are " + str(labels)
    print "col labels are " + str(labels)
    print mat


"""
This method will load 3 kind of file
Seperable.csv, 3percent-miscategorization.csv,
10percent-miscatergorization.csv
And print the accuracy
"""


def run_files(file_name, k_val):

    if file_name == 'Seperable.csv':
        print "\nLoad Seperable.csv and train KNN with k ="+str(k_val)

    elif file_name == '3percent-miscategorization.csv':
        print "\nLoad 3percent-miscategorization.csv and train KNN with k ="+str(k_val)

    elif file_name == '10percent-miscatergorization.csv':
        print "\nLoad 10percent-miscategorization.csv and train KNN with k ="+str(k_val)

    # get the dataset from the csv file
    try:
        hw_data = np.genfromtxt(file_name, delimiter=',')
    except IOError:
        print "\nFail to load"+file_name
        print "Please check if the file is under the same directory."

    print "\nFinish loading " + file_name
    print "\npre-processing the data"
    # get the number of feature and test size
    hw_total_size = hw_data.shape[0]
    hw_test_size = int(hw_total_size * 0.1)
    random.seed(1)

    testing_sample_index = select_test_sample_index(hw_test_size, hw_total_size, random)
    training_sample_index = []

    for i in range(hw_total_size):
        if i not in testing_sample_index:
            training_sample_index.append(i)

    # extract the testing and training sample from the dataset
    testing_sample = hw_data[testing_sample_index, :]
    training_sample = hw_data[training_sample_index, :]

    # apply the z-transform to the testing dataset and training dataset
    # with the mean and standard deviation of training dataset
    z_trans_result = z_transform(testing_sample, training_sample)

    z_trans_train = z_trans_result[0]
    z_trans_test = z_trans_result[1]

    train_X = z_trans_train[:, 0:9]
    train_Y = z_trans_train[:, 9]
    test_X = z_trans_test[:, 0:9]
    test_Y = z_trans_test[:, 9]
    train_Y = train_Y.astype(int)

    predict_Y = KNN(train_X, train_Y, test_X, k_val)

    acc = accuracy(predict_Y, test_Y)

    print "\naccuracy is " + str(acc)

    confusion_matrix(predict_Y, test_Y)


def load_abalone(k_val):
    print "\nLoad abalone csv file and train KNN with k =" + str(k_val)
    # get the dataset from the csv file
    abalone_data = np.genfromtxt('abalone.csv', delimiter=',', dtype=None)
    abalone_data2 = np.genfromtxt('abalone.csv', delimiter=',')
    num_part = np.delete(abalone_data2, 0, 1)

    # get the number of feature and test size
    abalone_total_size = abalone_data.shape[0]
    abalone_test_size = int(abalone_total_size * 0.1)
    random.seed(1)

    # transform the categorical column into one-hot encoding
    #  numerical column
    categ = get_column(abalone_data, 0)
    tmp_categ = categ_to_numerical(categ)
    new_categ = one_hot_encoding(tmp_categ)
    new_dataset = np.hstack((new_categ, num_part))

    testing_sample_index = select_test_sample_index(abalone_test_size, abalone_total_size, random)
    training_sample_index = []

    for i in range(abalone_total_size):
        if i not in testing_sample_index:
            training_sample_index.append(i)

    # extract the testing and training sample from the dataset
    testing_sample = new_dataset[testing_sample_index, :]
    training_sample = new_dataset[training_sample_index, :]

    # apply the z-transform to the testing dataset and training dataset
    # with the mean and standard deviation of training dataset
    z_trans_result = z_transform(testing_sample, training_sample)

    z_trans_train = z_trans_result[0]
    z_trans_test = z_trans_result[1]

    train_X = z_trans_train[:, 0:10]
    train_Y = z_trans_train[:, 10]
    test_X = z_trans_test[:, 0:10]
    test_Y = z_trans_test[:, 10]

    train_Y = train_Y.astype(int)
    predict_Y = KNN(train_X, train_Y, test_X, k_val)

    acc = accuracy(predict_Y, test_Y)

    print "\naccuracy is " + str(acc)
    confusion_matrix(predict_Y, test_Y)

"""
main function loads abalone.csv under the same folder

"""

if __name__ == "__main__":

    list_k = [1, 3, 5, 7, 9]

    for k in list_k:
        load_abalone(k)

    for k in list_k:
        run_files('Seperable.csv', k)

    for k in list_k:
        run_files('3percent-miscategorization.csv', k)

    for k in list_k:
        run_files('10percent-miscatergorization.csv', k)
