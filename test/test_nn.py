import numpy as np
from nn import nn, preprocess

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
    my_nn = nn.NeuralNetwork(nn_arch=GENERIC_ARCH, seed=42, lr=1, batch_size=500, epochs=1000, loss_function='bin_ce')
    y = np.random.randint(0, high=2, size=5)
    y_hat = np.random.random(size=5)
    assert my_nn._binary_cross_entropy(y, y_hat) == -np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / y.shape[0]

def test_binary_cross_entropy_backprop():
    """
    Tests that the _binary_cross_entropy_backprop() method returns correct values.
    """
    my_nn = nn.NeuralNetwork(nn_arch=GENERIC_ARCH, seed=42, lr=1, batch_size=500, epochs=1000, loss_function='bin_ce')
    y = np.random.randint(0, high=2, size=5)
    y_hat = np.random.random(size=5)
    assert np.isclose(my_nn._binary_cross_entropy_backprop(y, y_hat), -(y - y_hat) / ((1-y_hat) * y_hat * y.shape[0] + 1e-10))

def test_mean_squared_error():
    """
    Tests that the _mean_squared_error() loss method returns correct values.
    """
    my_nn = nn.NeuralNetwork(nn_arch=GENERIC_ARCH, seed=42, lr=1, batch_size=500, epochs=1000, loss_function='mse')
    y = np.random.randint(0, high=2, size=5)
    y_hat = np.random.random(size=5)
    assert my_nn._mean_squared_error(y, y_hat) == np.sum((y - y_hat) ** 2) / len(y)

def test_mean_squared_error_backprop():
    """
    Tests that the _mean_squared_error_backprop() method returns correct values.
    """
    my_nn = nn.NeuralNetwork(nn_arch=GENERIC_ARCH, seed=42, lr=1, batch_size=500, epochs=1000, loss_function='mse')
    y = np.random.randint(0, high=2, size=5)
    y_hat = np.random.random(size=5)
    assert np.isclose(my_nn._mean_squared_error_backprop(y, y_hat), -(y - y_hat) / y.shape[0])

def test_sample_seqs():
    """
    Tests that the sample_seqs() function correctly upsamples DNA sequences from the smaller of two
    input classes, as needed in this assignment.
    """
    seqs = ['AAA', 'GGG', 'CCC', 'TT', 'TT']
    labels = [False, False, False, True, True]
    resampled_seqs, resampled_labels = preprocess.sample_seqs(seqs, labels)

    subseqs = set([seq[:2] for seq in resampled_seqs] + [seq[1:] for seq in resampled_seqs])
    for seq in seqs:
        # check that max length seqs are still there
        if len(seq) == 3:
            assert seq in resampled_seqs

        # check that shorter seqs were correctly padded
        else:
            assert seq in subseqs

    # check that true seqs were upsampled and same number of false sequences are present
    assert resampled_labels.count(True) == 3
    assert resampled_labels.count(False) == 3

    # check that positive and negative labels are still matched with correct sequences
    for seq, label in zip(resampled_seqs, resampled_labels):
        if label == False:
            assert seq in ('AAA', 'GGG', 'CCC')

def test_one_hot_encode_seqs():
    """
    Tests that the one_hot_encode_seqs() function correctly encodes DNA sequences into 1-hot vectors.
    """
    seqs = ['AT', 'GC']
    seqs_1hot = [np.array([1, 0, 0, 0, 0, 1, 0, 0]), np.array([0, 0, 0, 1, 0, 0, 1, 0])]
    assert np.array(preprocess.one_hot_encode_seqs(seqs)).all() == np.array(seqs_1hot).all()