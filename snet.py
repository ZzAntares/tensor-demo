from sknn.mlp import Classifier, Layer
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
X_train = pd.read_csv(os.path.join(BASE_DIR, 'keys.csv'))
y_train = pd.read_csv(os.path.join(BASE_DIR, 'values.csv'))

from sknn.mlp import Regressor, Layer

nn = Regressor(
    layers=[
        Layer("Input", units=4),
        Layer("Hidden", units=5),
        Layer("Output", units=5),
    ],
    learning_rate=0.02,
    n_iter=10)

nn.fit(X_train, y_train)

# prediction
y_valid = nn.predict(X_valid)

score = nn.score(X_test, y_test)
