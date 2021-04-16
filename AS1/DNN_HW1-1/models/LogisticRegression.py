import numpy as np

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.random.rand(self.num_features, 1)

    def train(self, x, y, epochs, batch_size, lr, optim):
        loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses
        # Weights are updated through the optimizer, not directly within 'train' function.

        # Tip : log computation may cause some error, so try to solve it by adding an epsilon(small value) within log term.
        epsilon = 1e-7
        # ========================= EDIT HERE ========================
        y = y.reshape(x.shape[0], 1)
        w = self.W

        for i in range(epochs):
            loss = 0
            for j in range(0, x.shape[0], batch_size):
                wd = np.zeros_like(self.W)

                x_batch = x[j:j+batch_size]
                y_batch = y[j:j+batch_size]
                
                y_predicted = np.dot(x_batch, self.W)
                y_predicted = self._sigmoid(y_predicted)

                cost = -y_batch * np.log(y_predicted + epsilon) - (1 - y_batch) * np.log(1 - y_predicted + epsilon)
                loss = cost.mean()
                wd = np.dot(x_batch.transpose(), y_predicted - y_batch) / x_batch.shape[0]
                self.W = optim.update(self.W, wd, lr)
        # ============================================================
        return loss

    def forward(self, x):
        threshold = 0.5
        y_predicted = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # The model predicts the label as 1 if the probability is greater or equal to 'threshold'
        # Otherwise, it predicts as 0

        # ========================= EDIT HERE ========================
        y_predicted = np.dot(x, self.W)
        y_predicted = LogisticRegression._sigmoid(self, y_predicted)
        y_predicted = np.array([1 if p >= threshold else 0 for p in y_predicted]).reshape(-1, 1)

        # ============================================================

        return y_predicted

    def _sigmoid(self, x):
        sigmoid = None

        # Sigmoid Function
        # The function returns the sigmoid of 'x'

        # ========================= EDIT HERE ========================
        sigmoid = 1/(1 + np.exp(-x)) 
        # ============================================================
        return sigmoid
