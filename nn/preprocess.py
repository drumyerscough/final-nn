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
    lbl_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    
    # assume number of sequences to inc
    pos_seqs = []
    neg_seqs = []
    for label, seq in zip(labels, seqs):
        if label:
            pos_seqs.append(seq)
        elif not label:
            neg_seqs.append(seq)
    
    num_pos = len(pos_seqs)
    num_neg = len(neg_seqs)
    sampled_seqs = []
    sampled_labels = []
    for idx in range(max(num_pos, num_neg)):
        rand_idx = np.random.randint(num_pos)
        rand_resi_to_mutate = np.random.randint(len(pos_seqs[rand_idx]))
        mutated_seq = pos_seqs[rand_idx].copy()
        mutated_seq[rand_resi_to_mutate] = lbl_map[np.random.randint(4)]
        sampled_seqs.append(mutated_seq)
        sampled_seqs.append(True)

        rand_idx = np.random.randint(num_neg)
        sampled_seqs.append(neg_seqs[rand_idx])

        sampled_labels.append(False)

    sampled_array = np.column_stack((sampled_seqs, sampled_labels))

    return sampled_array[:, 0].tolist(), sampled_array[:, 1].tolist()


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
    seq_idxs = list(map(lambda x: lbl_map[x], seq_arr))
    return np.eye(len(lbl_map))[seq_idxs].flatten(order='C')