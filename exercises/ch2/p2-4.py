import numpy as np

# Define the autocorrelation matrix R and the vector r
R = np.array([[0.3040, 0.3033],
              [0.3033, 0.3040]])
r = np.array([0.3033])

R_1 = R[0, 0]  # order1
a_1 = r[0] / R_1

print("Coefficients of the first-order minimum variance linear predictor (a_1):", a_1)
