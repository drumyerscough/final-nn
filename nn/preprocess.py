# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    lbl_map = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
    mapper = np.vectorize(pyfunc=lambda x: lbl_map[x])
    
    # divide inputs into positive and negative examples
    pos_seqs = []
    neg_seqs = []
    for label, seq in zip(labels, seqs):
        if label:
            pos_seqs.append(seq)
        elif not label:
            neg_seqs.append(seq)

    # find max sequence length
    max_seq_len = np.max([len(seq) for seq in seqs])

    # drop negative sequences with length less than max_seq_len
    # this loses some training data, but only 5 sequences
    neg_seqs = [seq for seq in neg_seqs if len(seq) == max_seq_len]
    
    # upsample the smaller class to the size of the larger class
    num_pos = len(pos_seqs)
    num_neg = len(neg_seqs)
    sampled_seqs = []
    sampled_labels = []
    for idx in range(max(num_pos, num_neg)):
        # pick a random Rap1 binding seq from the positives
        rand_idx = np.random.randint(num_pos)
        mutated_seq = pos_seqs[rand_idx]

        # originally I considered making mutations to the Rap1 motifs, but I decided it wasn't necessary
        #rand_resi_to_mutate = np.random.randint(len(pos_seqs[rand_idx]))
        #mutated_seq = mutated_seq[:rand_resi_to_mutate] + lbl_map[np.random.randint(4)] + mutated_seq[rand_resi_to_mutate+1:]

        # generate a random DNA sequence of length max_seq_len and insert the Rap1 motif at a random position
        rand_idx = np.random.randint(max_seq_len - num_pos)
        padded_seq = ''.join(mapper(np.random.randint(0, high=4, size=max_seq_len)).tolist())
        padded_seq = padded_seq[:rand_idx] + mutated_seq + padded_seq[rand_idx+len(mutated_seq):]

        sampled_seqs.append(padded_seq)
        sampled_labels.append(True)

        # add a negative sequence (no resampling)
        sampled_seqs.append(neg_seqs.pop())
        sampled_labels.append(False)

    # could shuffle here, but I won't since I'll use sklearn for train/test splitting, which shuffles

    return sampled_seqs, sampled_labels


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    lbl_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    seq_idxs = list(map(lambda x: 
                        list(map(lambda y: lbl_map[y], x)), 
                        seq_arr))

    encodings = []
    for seq in seq_idxs:
        encodings.append(np.eye(4)[seq].flatten(order='C'))

    return encodings