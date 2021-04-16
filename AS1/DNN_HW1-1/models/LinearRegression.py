import numpy as np

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch_size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses
        # Weights are updated through the optimizer, not directly within 'train' function.
        # ========================= EDIT HERE ========================
        y = y.reshape(x.shape[0], 1)
        w = self.W

        for i in range(epochs):
            for j in range(0, x.shape[0], batch_size):
                wd = np.zeros_like(self.W)

                x_batch = x[j:j+batch_size]
                y_batch = y[j:j+batch_size]

                y_predicted = self.forward(x_batch)
                final_loss = (pow(y_batch - y_predicted, 2)).mean()
                wd = 2 * np.dot(x_batch.transpose(), (y_predicted - y_batch)) / x_batch.shape[0]

                w = optim.update(w, wd, lr)
                
                self.W = w
        # ============================================================
        return final_loss

    def forward(self, x):
        y_predicted = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'

        # ========================= EDIT HERE ========================
        y_predicted = np.dot(x, self.W)
        # ============================================================
        return y_predicted
