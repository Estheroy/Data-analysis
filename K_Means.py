import numpy as np
import random
import operator
from heapq import heappop
from heapq import heappush
import matplotlib.pyplot as plt
import csv
import time


def append_ones(m):
    return np.concatenate((m, np.ones((m.shape[0], 1))), axis=1)


"""
Given two data points with n features

Return: the euclidian distance between these two data points
"""


def get_euclidian_distance(x1, x2):

    dist = np.zeros((x1.shape[0], 1))
    for i in range(x1.shape[0]):
        dist[i] = np.sqrt(sum(np.square(x1[i,:] - x2[i,:])))

    return dist

def get_squared_euclidian_distance(x1, x2):

    dist = np.zeros((x1.shape[0], 1))
    for i in range(x1.shape[0]):
        dist[i] = sum(np.square(x1[i,:] - x2[i,:]))

    return dist


"""
KMeans algorithm to predict label
"""

def KMeans(X, init_k):

    random.seed(2)
    print "\nTrain KMeans with init k = " + str(init_k)
    random_index = random.sample(range(0, X.shape[0]), init_k)

    # generate K number of samples from dataset
    centers = X[random_index, :]
    prev_centers =  X[random_index, :]

    # initialize the error of change
    dist = np.zeros((X.shape[0], init_k))

    WCSS_error = 0
    cluster_index = 0

    while (1):

        if init_k == 1:
            temp_center = np.tile(centers, (X.shape[0], 1))
            dist = get_euclidian_distance(X, temp_center)
        else:
            # compute the distance
            for i in range(init_k):
                temp_center = np.tile(centers[i, :], (X.shape[0], 1))
                dist[:, i] = get_euclidian_distance(X, temp_center)[:,0]

        index = np.argmin(dist, axis=1)
        cluster_index = index

        for i in range(init_k):

            centroid_index = np.where(index == i)
            centroid_point = np.reshape(X[centroid_index, :],(X[centroid_index, :].shape[1], -1))
            centers[i, :] = np.mean(centroid_point, axis=0)

        change = np.sum(np.absolute(prev_centers - centers))
        prev_centers = centers

        if change == 0:

            for i in range(init_k):

                centroid_index = np.where(index == i)
                centroid_point = np.reshape(X[centroid_index, :], (X[centroid_index, :].shape[1], -1))
                # repmat the center points
                temp_center = np.tile(centers[i, :], (X[centroid_index, :].shape[1], 1))
                WCSS_error = WCSS_error + np.sum(get_squared_euclidian_distance(centroid_point, temp_center))
            break

    print "Mean Value of Each Feature is:"

    for i in range(init_k):
        centroid_index = np.where(index == i)
        centroid_point = np.reshape(X[centroid_index, :], (X[centroid_index, :].shape[1], -1))
        print np.mean(centroid_point, axis=0)

    print "Standard Deviation of Each Feature is:"
    for i in range(init_k):
        centroid_index = np.where(index == i)
        centroid_point = np.reshape(X[centroid_index, :], (X[centroid_index, :].shape[1], -1))
        print np.std(centroid_point, axis=0)

    return WCSS_error, centers, cluster_index


"""
main function loads abalone.csv under the same folder

"""

if __name__ == "__main__":

    ################### Part0: Pre-Processing ###################
    # load data
    with open('abalone.csv', 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = []
        for row in reader:
            s = row[0]
            sex = [1, 0, 0] if s == 'F' else ([0, 1, 0] if s == 'M' else [0, 0, 1])
            data.append(sex + [float(r) for r in row[1:]])
        data = np.array(data)

    # create train (90%) and test set (10%)
    np.random.seed(0)
    np.random.shuffle(data)
    train_num = int(data.shape[0] * 0.9)
    X_train = data[:train_num, :-1]
    Y_train = data[:train_num, -1]
    X_test = data[train_num:, :-1]
    Y_test = data[train_num:, -1]

    # Remove the first three sex column
    X_train_QR = X_train[:, 3:11]
    Y_train_QR = Y_train
    X_test_QR = X_test[:, 3:11]
    Y_test_QR = Y_test

    ###########################################
    # z-scale
    X_means = np.mean(X_train, axis=0)
    X_stds = np.std(X_train, axis=0)
    X_train = (X_train - X_means) / X_stds
    X_test = (X_test - X_means) / X_stds

    # append ones
    X_train = append_ones(X_train)
    X_test = append_ones(X_test)

    # Here I used the enabled z-scale and enabled append ones dataset
    ###########################################


    ################### Part1: K-Means ###################
    print "Part1: K-Means"

    # Need to change this manully othwise error
    # because I didn't reset the RMSE in each loop

    K = [1,2,4,8,16]
    count = 0
    for i in K:
        result = KMeans(X_train, i)
        model_centroids = result[1]
        print "Centroids are"
        print model_centroids
        print "WCSS is " + str(result[0])

        ################### Part2: K-Means and QR ###################
        test_centroid_index = np.zeros((i, 1))
        dist = np.zeros((X_train.shape[0], i))
        dist2 = np.zeros((X_test.shape[0], i))

        for iter in range(i):
            temp_center = np.tile(model_centroids[iter, :], (X_train.shape[0], 1))
            dist[:, iter] = get_euclidian_distance(X_train, temp_center)[:, 0]
            temp_center = np.tile(model_centroids[iter, :], (X_test.shape[0], 1))
            dist2[:, iter] = get_euclidian_distance(X_test, temp_center)[:, 0]

        train_index = np.argmin(dist, axis=1) # the cluster index of each datapoint in X_test_QR
        test_index = np.argmin(dist2, axis=1)

        index = result[2]  # cluster index # not used
        cluster_sum = np.array([])

        # i is the value of k here
        for j in range(i):

            centroid_index = np.where(train_index == j)
            centroid_index_test = np.where(test_index == j)

            centroid_point_train_X = (X_train_QR[centroid_index, :])[0,:,:]
            centroid_point_train_Y = Y_train_QR[centroid_index]

            beta, _, _, _ = np.linalg.lstsq(centroid_point_train_X, centroid_point_train_Y)

            centroid_point_test_X = (X_test_QR[centroid_index_test, :])
            centroid_point_test_Y = Y_test_QR[centroid_index_test]

            cluster_sum = np.append(cluster_sum, (np.dot(centroid_point_test_X, beta) - centroid_point_test_Y) ** 2)

        print "RMSE is " + str(np.sqrt(np.mean((cluster_sum))))

        # linear least square Y = X beta
        beta, _, _, _ = np.linalg.lstsq(X_train, Y_train)

        # root mean square error
        print "Overall RMSE is " + str(np.sqrt(np.mean((np.dot(X_test, beta) - Y_test) ** 2)))

        count = count + 1