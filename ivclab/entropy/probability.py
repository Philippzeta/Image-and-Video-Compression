import numpy as np
from einops import rearrange
from numpy.lib.stride_tricks import sliding_window_view

def stats_joint(image, pixel_range):
    """
    Computes joint probability of non-overlapping horizontal pixel pairs
    of an image, similar to stats_marg function. However, this
    counts every instance of pixel pairs in a 2D table to
    find the frequencies. Then, it normalizes the values to
    convert them to probabilities. Return a 1D vector
    since this is how we represent pmf values.

    Hint: You can use np.histogram2d for counting quickly over pixel pairs

    image: np.array of shape [H, W, C]
    pixel_range: np.array of shape [B] where B is number of bins, e.g. pixel_range=np.arange(256)

    returns 
        pmf: np.array of shape [B^2], probability mass function of image pixel pairs over range
    """
    # A table to hold count of pixel pair occurences
    count_table = np.zeros((len(pixel_range), len(pixel_range)))

    # Get all non overlapping horizontal pixel pairs as an array of shape [N, 2]
    pixel_pairs = rearrange(image, 'h (w s) c -> (h w c) s', s=2)

    # YOUR CODE STARTS HERE
    pixel_pairs = pixel_pairs.astype(float)

    # Use np.histogram2d to calculate the joint histogram
    counts, x_edges, y_edges = np.histogram2d(pixel_pairs[:, 0], pixel_pairs[:, 1], bins=[pixel_range, pixel_range])

    # Normalize to get the joint PMF
    pmf = counts.flatten() / np.sum(counts)
    # YOUR CODE ENDS HERE
    return pmf

def stats_cond(image, pixel_range, eps=1e-8):
    """
    Computes conditional probability of overlapping horizontal pixel pairs
    of an image, similar to stats_joint function. The conditional probability
    is found by the formula SUM{ - p(x,y) * ( log2( p(x,y) ) - log2( p(x) ) ) }. To compute
    p(x), you can take the sum of normalized probabilities of p(x,y) over row axis.
    Make sure to add a small epsilon before computing the log probabilities. You can
    ignore the first pixels in every row since they don't have a left neighbor.

    Hint: You can use np.histogram2d for counting quickly over pixel pairs

    image: np.array of shape [H, W, C]
    pixel_range: np.array of shape [B] where B is number of bins, e.g. pixel_range=np.arange(256)

    returns 
        cond_entropy: a scalar value
    """
    # A table to hold count of pixel pair occurences
    pmf_table = np.zeros((len(pixel_range), len(pixel_range)))

    # Get all overlapping horizontal pixel pairs as an array of shape [N, 2]
    pixel_pairs = rearrange(sliding_window_view(image, 2, axis=1), 'h w c s-> (h w c) s', s=2) 

    # YOUR CODE STARTS HERE
    # Convert pixel pairs to float for histogram calculation
    pixel_pairs = pixel_pairs.astype(float)

    # Calculate the joint histogram using np.histogram2d
    counts, _, _ = np.histogram2d(pixel_pairs[:, 0], pixel_pairs[:, 1], bins=[pixel_range, pixel_range])

    # Normalize to get joint PMF
    joint_pmf = counts / np.sum(counts)

    # Calculate marginal PMF for the first pixel in the pair (p(x))
    marginal_pmf = np.sum(joint_pmf, axis=1)

    # Add a small epsilon to avoid log(0)
    joint_pmf += eps
    marginal_pmf += eps

    # Calculate conditional entropy
    cond_entropy = -np.sum(joint_pmf * (np.log2(joint_pmf) - np.log2(marginal_pmf[:, None])))

    # YOUR CODE ENDS HERE
    return cond_entropy