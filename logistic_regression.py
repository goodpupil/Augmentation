import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.utils import shuffle

mnist = fetch_mldata('MNIST original')
X = mnist.data.astype('float64')
y = mnist.target.astype('int')

# Binary classification, 4 vs 9
X = X[np.logical_or(y == 4, y == 9)]
y = y[np.logical_or(y == 4, y == 9)]

train_samples = 1000
test_samples = 5000
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=test_samples, random_state=0)
# Scale data to have mean zero and variance 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear logistic regression
# C controls regularization, which should be scaled by number of data points
lr = LogisticRegression(C=50 / train_samples)
lr.fit(X_train, y_train)
print('Linear logistic regression acc.: {}'.format(lr.score(X_test, y_test)))

# Kernel logistic regression with RFF feature map
num_features = 1000
rff_map = RBFSampler(gamma=0.001, random_state=0, n_components=num_features)
X_train_transformed = rff_map.fit_transform(X_train)
X_test_transformed = rff_map.transform(X_test)
lr = LogisticRegression(C=50 / train_samples)
lr.fit(X_train_transformed, y_train)
print('Kernel logistic regression acc.: {}'.format(
    lr.score(X_test_transformed, y_test)))
w_kernel_lr = lr.coef_

# Explicit data augmentation (rotation between -15 and 15 degrees)
angles = range(-15, 16, 2)
background_value = np.median(X_train)
rotate = lambda X, angle: ndimage.rotate(X.reshape((-1, 28, 28)),
    angle,
    axes=(2, 1),
    reshape=False,
    cval=background_value).reshape((-1, 28 * 28))
X_train_rotated = np.vstack(rotate(X_train, angle) for angle in angles)
y_train_rotated = np.tile(y_train, len(angles))

# Linear logistic regression + data augmentation
lr = LogisticRegression(C=50 / train_samples / len(angles))
lr.fit(X_train_rotated, y_train_rotated)
print('Linear logistic regression (with data augmentation) acc.: {}'.format(
    lr.score(X_test, y_test)))

# Kernel logistic regression with RFF feature map + data augmentation
X_train_rotated_transformed = rff_map.transform(X_train_rotated)
lr = LogisticRegression(C=50 / train_samples / len(angles))
lr.fit(X_train_rotated_transformed, y_train_rotated)
print('Kernel logistic regression (with data augmentation) acc.: {}'.format(
    lr.score(X_test_transformed, y_test)))
w_data_augmentation = lr.coef_.squeeze()

# Augmentation in the feature space by averaging
X_train_augmented = np.mean(
    X_train_rotated_transformed.reshape((len(angles), train_samples, -1)),
    axis=0)
lr = LogisticRegression(C=50 / train_samples)
lr.fit(X_train_augmented, y_train)
print('Kernel logistic regression (with feature augmentation) acc.: {}'.format(
    lr.score(X_test_transformed, y_test)))
w_feature_augmentation = lr.coef_.squeeze()

print(
    'Relative error between w_data_augmentation and w_feature_augmentation: {}'.
    format(
        np.linalg.norm(w_data_augmentation - w_feature_augmentation) /
        np.linalg.norm(w_data_augmentation)))
print(
    'Relative error between w_data_augmentation and w_feature_augmentation: {}'.
    format(
        np.linalg.norm(w_data_augmentation - w_kernel_lr) /
        np.linalg.norm(w_data_augmentation)))
