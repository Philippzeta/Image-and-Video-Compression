import numpy as np

def stats_marg(image, pixel_range):

    # Convert image to floating point
    image = image * 1.0

    # YOUR CODE STARTS HERE

    image_flattened = image.flatten()

    # Calculate histogram
    counts, _ = np.histogram(image_flattened, bins=pixel_range)

    # Normalize to get PMF
    pmf = counts / counts.sum()
    # YOUR CODE ENDS HERE
    return pmf

def calc_entropy(pmf, eps=1e-8):

    # It's good practice to add small epsilon
    # to get rid of bins with zeroes before taking logarithm
    pmf = pmf + eps
    
    # YOUR CODE STARTS HERE
    entropy = -np.sum(pmf * np.log2(pmf))
    # YOUR CODE ENDS HERE
    return entropy

def min_code_length(target_pmf, common_pmf, eps=1e-8):

    # It's good practice to add small epsilon
    # to get rid of bins with zeroes before taking logarithm
    common_pmf = common_pmf + eps
    
    # YOUR CODE STARTS HERE
    code_length = -np.sum(target_pmf * np.log2(common_pmf))
    # YOUR CODE ENDS HERE
    return code_length