import numpy as np
from nn import nn, io, preprocess

GENERIC_ARCH = [{'input_dim': 4000, 'output_dim': 250, 'activation': 'relu'}, 
                {'input_dim': 250, 'output_dim': 2, 'activation': 'sigmoid'}]

def test_single_forward():
    pass

def test_forward():
    pass

def test_single_backprop():
    pass

def test_predict():
    pass

def test_binary_cross_entropy():
    """
    Tests that the _binary_cross_entropy() loss method returns correct values.
    """
    my_nn = nn.NeuralNetwork(nn_arch=GENERIC_ARCH, lr=1, batch_size=500, epochs=1000, loss_function='mse')
    y = np.random.randint(0, high=2, size=5)
    y_hat = np.random.rand(0, high=1, size=5)
    assert my_nn._binary_cross_entropy() == -np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / y.shape[0]

def test_binary_cross_entropy_backprop():
    """
    Tests that the _binary_cross_entropy_backprop() method returns correct values.
    """
    my_nn = nn.NeuralNetwork(nn_arch=GENERIC_ARCH, lr=1, batch_size=500, epochs=1000, loss_function='mse')
    y = np.random.randint(0, high=2, size=5)
    y_hat = np.random.random(size=5)
    assert my_nn._mean_squared_error() == np.sum((y - y_hat) ** 2) / len(y)

def test_mean_squared_error():
    """
    Tests that the _mean_squared_error() loss method returns correct values.
    """
    my_nn = nn.NeuralNetwork(nn_arch=GENERIC_ARCH, lr=1, batch_size=500, epochs=1000, loss_function='mse')
    y = np.random.randint(0, high=2, size=5)
    y_hat = np.random.randint(0, high=2, size=5)
    assert my_nn._mean_squared_error() == np.sum((y - y_hat) ** 2) / len(y)

def test_mean_squared_error_backprop():
    """
    Tests that the _mean_squared_error_backprop() method returns correct values.
    """

def test_sample_seqs():
    """
    Tests that the sample_seqs() function correctly upsamples DNA sequences from the smaller of two
    input classes, as needed in this assignment.
    """

def test_one_hot_encode_seqs():
    """
    Tests that the one_hot_encode_seqs() function correctly encodes DNA sequences into 1-hot vectors.
    """
