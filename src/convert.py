import numpy as np
from operator import itemgetter

### Functions for individual votes

def dict_to_strict_rank_order(vote_dict, check=True):
    """
    Convert a vote of the form {candidate: rank, ...} (1 best, 2 second-best, etc.)
    to [candidate1, candidate2, ...] (best first)
    :param vote_dict: vote as dictionary
    :param check: make sure that all candidates have different ranks
    :return: vote as ranked list
    """
    if check and len(set(vote_dict.values())) != len(vote_dict):
        raise ValueError('Each candidate must have a different rank')
    sorted_items = sorted(vote_dict.items(), key=itemgetter(1))
    return [candidate for candidate, _ in sorted_items]

def dict_to_rank_order(vote_dict):
    """
    Convert a vote of the form {candidate: rank, ...} (1 best, 2 second-best, etc.)
    to [{set of candidates 1}, {set of candidates 2, ...}] (best first)
    :param vote_dict: vote as dictionary
    :return: vote as ranked list of tied sets
    """
    inverse_dict = dict()
    for cand, rank in vote_dict.items():
        inverse_dict.setdefault(rank, set()).add(cand)
    return [inverse_dict[r] for r in sorted(inverse_dict)]
    

### Functions for whole elections

def partial_strict_orders_to_pref_mat(votes, candidates=None):
    """
    Convert a list of partial strict rank orders to a preference matrix,
    where pref_mat[i,j] is the number of voters preferring i to j
    :param votes: list of lists (each vote is a rank order)
    :param candidates: list of candidates
    :return: preference matrix
    """
    # Collect candidates if not specified
    if candidates == None:
        candidates = sorted({x for v in votes for x in v})
    # Map from candidate names to indices
    ind = {name: i for i, name in enumerate(candidates)}
    # Initialise preference matrix
    N = len(candidates)
    pref_mat = np.zeros((N,N))
    # Collate votes
    for v in votes:
        for i, high in enumerate(v):
            for low in v[i+1:]:
                # Increase the count when one candidate is ranked after another
                pref_mat[ind[high], ind[low]] += 1
    return pref_mat, candidates

def partial_orders_to_pref_mat(votes, candidates=None):
    """
    Convert a list of partial rank orders to a preference matrix,
    where pref_mat[i,j] is the number of voters preferring i to j
    :param votes: list of lists of sets (each vote is a rank order with possible ties)
    :param candidates: list of candidates
    :return: preference matrix
    """
    # Collect candidates if not specified
    if candidates == None:
        candidates = sorted({x for v in votes for x_set in v for x in x_set})
    # Map from candidate names to indices
    ind = {name: i for i, name in enumerate(candidates)}
    # Initialise preference matrix
    N = len(candidates)
    pref_mat = np.zeros((N,N))
    # Collate votes
    for v in votes:
        for i, high_set in enumerate(v):
            for low_set in v[i+1:]:
                # All candidates in high_set are ranked above all in low_set
                for high in high_set:
                    for low in low_set:
                        # Increase the count when one candidate is ranked after another
                        pref_mat[ind[high], ind[low]] += 1
                    for other_high in high_set - {high}:
                        # Increase the count by 1/2 on both sides, for ties
                        pref_mat[ind[high], ind[other_high]] += 1/2
    return pref_mat, candidates

def normalise_pref_mat(pref_mat, smoothing=0.5):
    """
    Normalise a preference matrix from numbers of votes
    to proportions of votes, with optional smoothing
    :param pref_mat: absolute numbers of votes for each preference
    :param smoothing: (default 0.5) amount to add to each preference
    :return: normalised preference matrix, weight matrix
    """
    normed = np.copy(pref_mat)
    normed += smoothing
    sums = normed + np.transpose(normed)
    normed /= sums
    if smoothing == 0:  # In case of 0/0
        normed[np.isnan(normed)] = 1/2
    np.fill_diagonal(normed, 0)
    weight = pref_mat + np.transpose(pref_mat) + 2*smoothing
    return normed, weight
