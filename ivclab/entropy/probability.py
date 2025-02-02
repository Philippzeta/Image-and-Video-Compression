import numpy as np
from einops import rearrange
from numpy.lib.stride_tricks import sliding_window_view

def stats_joint(image, pixel_range):

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