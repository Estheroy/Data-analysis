import numpy as np
import random
from numpy.linalg import norm
from math import copysign
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


def OneHotEncoding(labels):

    num_class = len(np.unique(labels))
    arr = np.zeros((len(labels), num_class), dtype=np.int)
    for i in range(len(labels)):
        arr[i][labels[i]] = 1

    return arr

"""
Given a dataset

Return: n x 1 column which is the i th row of the dataset
"""


def get_column(data, i):
    return [row[i] for row in data]


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

#####################################################################################


def qr_decompose(A):
    """Perform QR decomposition of matrix A using Householder reflection."""
    (num_rows, num_cols) = np.shape(A)
    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(num_rows)
    R = np.copy(A)

    # Iterative over column sub-vector and
    # compute Householder matrix to zero-out lower triangular matrix entries.
    for cnt in range(num_cols):
        x = R[cnt:, cnt]
        e = np.zeros_like(x)
        e[0] = 1
        norm_x = copysign(np.linalg.norm(x), -R[cnt, cnt])
        v = np.dot(norm_x, e) - x

        Q_cnt = np.identity(num_rows)
        Q_cnt[cnt:, cnt:] -= (2.0 * np.outer(v, v))/np.dot(v, v)

        R = np.dot(Q_cnt, R)
        Q = np.dot(Q_cnt, Q)

    z, n = R.shape

    for i in range(0, z):
        for j in range(0, n):
            if i > j:
                R[i][j] = 0
    return Q.transpose(), R


def back_solve(x, y):
    n, d = x.shape
    beta = np.array([0]*d).astype(float)
    for i in reversed(range(0, d)):
        for k in range(i+1, d):
            beta[i] += x[i][k]*beta[k]
        beta[i] = (y[i]-beta[i])/x[i, i]

    return beta

"""
main function loads abalone.csv under the same folder
"""


def load_abalone():
    # get the dataset from the csv file
    abalone_data = np.genfromtxt('abalone.csv', delimiter=',', dtype=None)
    abalone_data2 = np.genfromtxt('abalone.csv', delimiter=',')
    num_part = np.delete(abalone_data2, 0, 1)

    print "\nload abalone.csv"
    print "\nprocessing the data"
    # get the number of feature and test size
    abalone_total_size = abalone_data.shape[0]
    abalone_test_size = int(abalone_total_size*0.1)
    random.seed(1)

    # transform the categorical column into one-hot encoding
    #  numerical column
    categ = get_column(abalone_data, 0)
    tmp_categ = categ_to_numerical(categ)
    new_categ = OneHotEncoding(tmp_categ)
    new_dataset = np.hstack((new_categ,num_part))

    testing_sample_index = select_test_sample_index(abalone_test_size, abalone_total_size, random)
    training_sample_index = []

    for i in range(abalone_total_size):
        if i not in testing_sample_index:
            training_sample_index.append(i)

    # extract the testing and training sample from the data set
    testing_sample = new_dataset[testing_sample_index, :]
    training_sample = new_dataset[training_sample_index, :]

    train_X = training_sample[:, 0:10]
    train_Y = training_sample[:, 10]

    test_X = testing_sample[:,  0:10]
    test_Y = testing_sample[:, 10]
    train_Y = train_Y.astype(int)

    Q, R = qr_decompose(train_X)
    beta = back_solve(R, np.dot(Q.T, train_Y))

    acc = np.sqrt(np.mean((np.dot(test_X, beta)-test_Y)**2))
    print("RMSE for abalone data set is "+str(acc))



