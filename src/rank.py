import numpy as np

# Preference-matrix based methods returning scores

def markov(pref_mat, smoothing_type='comparison', smoothing_strength=0.5):
    """
    Score candidates by defining a Markov chain that jumps to better candidates
    :param pref_mat: unnormalised preference matrix
    :param smoothing_type: value must be one of:
      'uniform' - fixed probability of jumping to a random candidate
      'comparison' - a random jump is weighted as a fixed number of comparisons
    :param smoothing_strength: value interpreted according to smoothing_type
    :return: scores of candidates
    """
    # Check the input matrix
    N,M = pref_mat.shape
    if M != N:
        raise ValueError('Preference matrix must be square')
    if pref_mat.min() < 0:
        raise ValueError('Preference matrix must be non-negative')
    
    # Convert the preference matrix to a (transposed) transition matrix
    if smoothing_type == 'comparison':
        comp_totals = pref_mat.sum(0) + pref_mat.sum(1) + smoothing_strength
        trans = pref_mat / comp_totals  # Broadcasting: iterate along 0th index, each 1-index vector in pref_mat is divided by comp_totals vector
        trans += smoothing_strength * np.reciprocal(comp_totals) / (N-1)
    
    elif smoothing_type == 'uniform':
        comp_totals = pref_mat.sum(0) + pref_mat.sum(1)
        trans = pref_mat / comp_totals * (1 - smoothing_strength)
        trans += smoothing_strength / (N-1)
        # If one of the totals was 0 (candidate not compared), stay or jump randomly
        trans[:, comp_totals == 0] = 1/2 / (N-1) * (1 - smoothing_strength)
    
    # The above code should have set all the non-diagonal terms of the (transposed) transition matrix
    np.fill_diagonal(trans, 0)
    np.fill_diagonal(trans, 1 - trans.sum(0))
    
    # Convert to linear equation of the form Ax = b
    A = trans - np.diag(np.ones(N))
    b = np.zeros(N)
    # These equations are linearly dependent;
    assert np.linalg.matrix_rank(A) == N-1
    # Enforce the sum to be 1
    A[-1] = 1
    b[-1] = 1
    
    return np.linalg.solve(A,b)

# An alternative way to define a Markov chain for partial orders would be to:
#   1. choose a voter who ranked the current candidate
#   2. choose a comparison that this voter made.
# However, this requires knowing more than just the preference matrix 
