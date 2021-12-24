import numpy as np
from scipy.special import log_softmax
from sklearn.metrics import accuracy_score
from optimizer import SteepestDescentOptimizer


class NeuralNetworkModule:
    def __init__(self, dims=[2, 2], init_scale=1.0):
        self.n_layers = len(dims) - 1
        self.d_in = dims[0]
        self.d_out = dims[-1]
        self.parameters = {}
        self.gradients = {}
        self.outputs = {}
        for l in range(self.n_layers):
            self.parameters[f"weight_{l}"] = np.random.normal(
                scale=init_scale, size=(dims[l + 1], dims[l])
            )
            self.parameters[f"bias_{l}"] = np.random.normal(
                scale=init_scale, size=(dims[l + 1], 1)
            )
            self.gradients[f"weight_{l}"] = np.random.normal(
                scale=init_scale, size=(dims[l + 1], dims[l])
            )
            self.gradients[f"bias_{l}"] = np.random.normal(
                scale=init_scale, size=(dims[l + 1], 1)
            )

    def forward(self, x):
        """
        x.shape = [batch_size * dimension]
        """
        if x.shape[1] != self.d_in:
            raise ValueError("'x' should be of shape [batch_size * dimension].")
        z = x.T  # [dimension * batch_size]
        self.outputs["post_act_0"] = x
        for l in range(self.n_layers):
            z = np.dot(self.parameters[f"weight_{l}"], z) + self.parameters[f"bias_{l}"]
            self.outputs[f"pre_act_{l}"] = z.T.copy()
            if l < self.n_layers - 1:
                z = self._relu(z)
                self.outputs[f"post_act_{l+1}"] = z.T.copy()  # [batch_size * n_outputs]
        out = log_softmax(z, axis=0)  # [n_classes * batch_size]
        return out.T  # [batch_size * n_classes]

    def backward(self, y, scores):
        """
        scores = [batch_size * n_classes]
        y.shape = [batch_size]
        """
        n, n_classes = scores.shape

        # Gradient of the loss with respect to output of current layer.
        y_one_hot = np.squeeze(np.eye(n_classes)[y.reshape(-1)])
        class_probs = np.exp(scores)
        d_loss_d_out = -(y_one_hot - class_probs).T
        for l in range(self.n_layers - 1, -1, -1):
            self.gradients[f"weight_{l}"] = np.dot(
                d_loss_d_out, self.outputs[f"post_act_{l}"]
            )
            self.gradients[f"bias_{l}"] = np.sum(d_loss_d_out, axis=1).reshape(-1, 1)

            d_loss_d_out = np.dot(self.parameters[f"weight_{l}"].T, d_loss_d_out)
            if l > 0:
                d_loss_d_out = (
                    self._d_relu(self.outputs[f"pre_act_{l-1}"]).T * d_loss_d_out
                )

    def _compute_loss(self, y, scores):
        """
        score.shape = [batch_size * n_classes]
        y.shape = [batch_size]
        """
        n = len(y)
        return -scores[np.arange(n), y].sum()

    def _d_relu(self, x):
        return (x >= 0.0).astype(float)

    def _relu(self, x):
        return np.clip(x, 0.0, None)

    def update_params(self, optimizer):
        for param in list(self.parameters.keys()):
            self.parameters[param] = optimizer.update(
                self.parameters[param], self.gradients[param]
            )


class NeuralNetworkClassifier:
    def __init__(
        self,
        batch_size=16,
        epochs=32,
        dims=[2, 2, 2],
        init_scale=1.0,
        optimizer=SteepestDescentOptimizer(stepsize=1e-4),
        tol=1e-4,
        verbose=False,
    ):
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.epochs = epochs
        self.nn = NeuralNetworkModule(dims=dims, init_scale=init_scale)
        self.verbose = verbose
        self.tol = tol

    def fit(self, X, y):
        """
        Run mini-batch stochastic gradient descent to fit cross entropy loss objective.
        """
        n, d = X.shape
        checkpoint = self.epochs // 10
        stop = False
        loss_prev = self.nn._compute_loss(y, self.nn.forward(X))

        L_logits = []
        L_01s = []

        for epoch in range(self.epochs):

            order = np.random.permutation(n)
            num_batch = n // self.batch_size

            for i in range(num_batch):

                if len(X) > 1000:
                    idx = np.random.randint(len(X), size=1000)
                else:
                    idx = np.arange(len(X))

                indices = order[i : min(i + self.batch_size, n)]
                x_batch = X[indices]
                y_batch = y[indices]

                # Forward propagate.
                scores = self.nn.forward(x_batch)

                # Backward propagate.
                self.nn.backward(y_batch, scores)

                # Update parameters.
                self.nn.update_params(self.optimizer)

                # Compute loss.
                y_pred = self.predict(X[idx])
                train_accuracy = accuracy_score(y[idx], y_pred)
                loss = self.nn._compute_loss(y[idx], self.nn.forward(X[idx]))

                L_logits.append(loss / len(idx))
                L_01s.append(1 - train_accuracy)

                # Check topping criterion.
                # if self._stop(loss, loss_prev):
                #     stop = True
                #     if self.verbose:
                #         print("Stopping criterion reached.")
                #     break

                # Record loss.
                loss_prev = loss

            # Compute training loss.
            if self.verbose and (epoch % checkpoint == 0):
                print(
                    "Epoch %d \t | cross entropy loss: %0.4f \t | train accuracy: %0.3f"
                    % (epoch, loss, train_accuracy)
                )

            if stop:
                break

        self.L_logits = np.array(L_logits)
        self.L_01s = np.array(L_01s)

        return self

    def _stop(self, loss, loss_prev):
        return loss_prev - loss <= self.tol * loss_prev

    def predict(self, X):
        scores = self.nn.forward(X)
        return np.argmax(scores, axis=1)

    def predict_proba(self, X):
        scores = self.nn.forward(X)
        return np.exp(scores)

    def predict_log_proba(self, X):
        return self.nn.forward(X)
