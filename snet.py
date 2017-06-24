from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
X_train = pd.read_csv(os.path.join(BASE_DIR, 'keys.csv'))
y_train = pd.read_csv(os.path.join(BASE_DIR, 'values.csv'))

x = np.arange(0.0, 1, 0.01).reshape(-1, 1)
y = np.sin(2 * np.pi * x).ravel()
print "tipo de X"
print type(x)
sys.exit(0)

nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

n = nn.fit(x, y)
test_x = np.arange(0.0, 1, 0.05).reshape(-1, 1)
test_y = nn.predict(test_x)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x, y, s=1, c='b', marker="s", label='real')
ax1.scatter(test_x,test_y, s=10, c='r', marker="o", label='NN Prediction')
plt.show()
