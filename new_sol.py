# Linear regression one shot solution method
# Problem part 1

import numpy as np

# Load data
training_images = np.load("/home/archit/Desktop/smile_data/trainingFaces.npy")
training_labels = np.load("/home/archit/Desktop/smile_data/trainingLabels.npy")
test_images = np.load("/home/archit/Desktop/smile_data/testingFaces.npy")
test_labels = np.load("/home/archit/Desktop/smile_data/testingLabels.npy")

# Design matrix
X = training_images
X = np.hstack((np.ones((X.shape[0], 1)), X))    # appending bias unit to all
# Class labels
y = training_labels
y = y.reshape(y.shape[0],1)

# Assign values to notations
m = X.shape[0]  # no of training examples
n = X.shape[1] - 1  # no of input features per image 

# Intialize parameters
theta = np.zeros((n+1, 1))    # weights
Lambda = 1E6    # regularization strength parameter
mul_mat = np.eye(n)
mul_mat = np.hstack((np.zeros((mul_mat.shape[0], 1)), mul_mat))
mul_mat = np.vstack((np.zeros((1, mul_mat.shape[1])), mul_mat))

# One shot solution to find weights
A_inv = np.linalg.pinv( (np.dot(np.transpose(X), X) + Lambda*mul_mat) )
B = np.dot(np.transpose(X), y)
theta = np.dot(A_inv, B)

# Computing unregularized cost function on training set
hypothesis = np.dot(X, theta)
J_unreg_train = (1/(2*m)) * (np.sum((hypothesis - y)**2))
print("Unregularized cost for training data = " + str(J_unreg_train))

# Computing unregularized cost function on testing set

X_t = test_images     # load test images
X_t = np.hstack((np.ones((X_t.shape[0], 1)), X_t))    # appending bias unit to all
y_t = test_labels     # load test labels
y_t = y_t.reshape(y_t.shape[0], 1)

m_t = X_t.shape[0]      # number of training examples
hypothesis = np.dot(X_t, theta)
J_unreg_test = (1/(2*m_t)) * (np.sum((hypothesis - y_t)**2))
print("\nUnregularized cost for test data = " + str(J_unreg_test))

# Note: As per problem requirement we show that there exists a +ve value for 
# Lambda, such that training cost is higher than testing cost 
