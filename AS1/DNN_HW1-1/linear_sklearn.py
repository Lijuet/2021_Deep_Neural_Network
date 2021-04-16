from utils import _initialize, optimizer
import sklearn
from sklearn.linear_model import LinearRegression

# 1. Choose DATA : Titanic / Digit
# ========================= EDIT HERE ========================
# DATA
DATA_NAME = 'Concrete'
# ============================================================

assert DATA_NAME in ['Concrete', 'Graduate']

# Load dataset, model and evaluation metric
train_data, test_data, _, metric = _initialize(DATA_NAME)
train_x, train_y = train_data

num_data, num_features = train_x.shape
print('# of Training data : ', num_data)
MSE = 0.0
# ========================= EDIT HERE ========================
# MODEL
model = LinearRegression()

# TRAIN
model.fit(train_x, train_y)

# EVALUATION
test_x, test_y = test_data
MSE =  sklearn.metrics.mean_squared_error(test_y, model.predict(test_x))
# ============================================================

print('MSE on Test Data : %.2f ' % MSE)
