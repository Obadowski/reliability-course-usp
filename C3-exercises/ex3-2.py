import numpy as np
from scipy.special import gamma

alpha = 10      # years
beta = 0.5      # adimensional

# A failure until 1 year
pr_1year = 1 - np.exp(-(1/alpha)**beta)

# A failure until 10 years
pr_10years = 1 - np.exp(-(10/alpha)**beta)

# MTTF expected
mttf = alpha * gamma((1+beta)/beta)

print("Probability of failure after 1 year:", 1 - pr_1year)
print("Probability of failure after 10 years:", 1 - pr_10years)
print("MTTF:", mttf)
