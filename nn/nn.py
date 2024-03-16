# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        activations = {'relu': self._relu, 'sigmoid': self._sigmoid}
        f = activations[activation]
        print(W_curr.shape)
        print(A_prev.shape)
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        return f(Z_curr), Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        # make cache with layer indices as keys and 2-tuples of (Z, A) as values
        # entry for key=0 represents input data, so activations = X and Z is none
        cache = {0: (None, X.T)}

        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1 # note that self._param_dict starts with index 1
            cache[layer_idx] = self._single_forward(self._param_dict['W' + str(layer_idx)], 
                                                    self._param_dict['b' + str(layer_idx)], 
                                                    cache[idx][1],
                                                    layer['activation'])
        
        output = cache[layer_idx][1]
        return output, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        f_primes = {'relu': self._relu_backprop, 'sigmoid': self._sigmoid_backprop}
        f_prime = f_primes[activation_curr]

        print(W_curr.shape)
        print(dA_curr.shape)
        print(Z_curr.shape)

        delta = f_prime(dA_curr, Z_curr)

        print(A_prev.shape)
        print(delta.shape)

        # Compute partial derivatives
        #dW_curr = A_prev.T @ delta
        #db_curr = delta
        #dA_prev = delta @ W_curr
        dW_curr = delta@A_prev.T / A_prev.shape[1]
        db_curr = np.sum(delta, axis=1, keepdims=True) / A_prev.shape[1]
        dA_prev = W_curr.T @ delta

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        grad_dict = {idx+1: {'dW': 0, 'db': 0} for idx in range(len(self.arch))}

        # set dA_prev for output layer based on ground truth labels
        loss_backprops = {'bin_ce': self._binary_cross_entropy_backprop, 'mse': self._mean_squared_error_backprop}
        loss_backprop = loss_backprops[self._loss_func]

        f_primes = {'relu': self._relu_backprop, 'sigmoid': self._sigmoid_backprop}
        f_prime = f_primes[self.arch[-1]['activation']] # takes inputs dA, Z

        #dA_curr = f_prime(loss_backprop(y, y_hat), cache[len(self.arch)][0])
        dA_curr = loss_backprop(y, y_hat)

        for idx in range(1,1+len(self.arch)):
            print('backpropping thru layer')
            layer = self.arch[-idx]
            layer_idx = len(self.arch) - idx + 1
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr=self._param_dict['W' + str(layer_idx-1)],
                                                         b_curr=self._param_dict['b' + str(layer_idx-1)], 
                                                         Z_curr=cache[layer_idx][0],
                                                         A_prev=cache[layer_idx-1][1],
                                                         dA_curr=dA_curr,
                                                         activation_curr=self.arch[layer_idx-1]['activation'])
            grad_dict[layer_idx]['dW'] += dW_curr
            grad_dict[layer_idx]['db'] += db_curr

            dA_curr = dA_prev

        for layer_idx, layer in grad_dict.items():
            layer['dW'] = layer['dW'] / y.shape[0]
            layer['db'] = layer['db'] / y.shape[0]
        
        self._update_params(grad_dict)

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        print('might be forgetting 1/m terms')
        alpha = self._lr
        for layer_idx, grad in grad_dict.items():
            self._param_dict['W' + str(layer_idx)] = self._param_dict['W' + str(layer_idx)] - alpha*grad['dW']
            self._param_dict['b' + str(layer_idx)] = self._param_dict['b' + str(layer_idx)] - alpha*grad['db']


    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        losses = {'bin_ce': self._binary_cross_entropy, 'mse': self._mean_squared_error}
        loss = losses[self._loss_func]

        num_batches = int(np.ceil(X_train.shape[0] / self._batch_size))
        for epoch in range(self._epochs):
            print(f'starting epoch #{epoch}')

            batches = np.concatenate((X_train, y_train), axis=1)
            np.random.shuffle(batches)
            for batch in np.array_split(batches, num_batches):
                # note that X should be a matrix with shape [batch_size, features]
                X_batch = batch[:, :X_train.shape[1]]
                y_batch = batch[:, X_train.shape[1]:]
                output, cache = self.forward(X_batch)
                grad_dict = self.backprop(y_batch.T, output, cache)

                per_epoch_loss_train.append(loss(output.T, y_batch))
                #per_epoch_loss_val.append(loss(self.predict(X_val), y_val))
            print(f'done')

        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        output, _ = self.forward(X)
        return output

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1/(1 + np.exp(Z))

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        f_Z = self._sigmoid(Z)
        return f_Z * (1 - f_Z)

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(Z, 0)

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """

        return np.greater(Z, 0) * dA

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        return -np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / y.shape[0]

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        return -(y - y_hat) / ((1-y_hat) * y_hat * y.shape[0])

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        return np.sum((y - y_hat) ** 2) / len(y)

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        return y - y_hat / len(y)
        #return np.sum(y_hat*(y - y_hat)) / len(y)