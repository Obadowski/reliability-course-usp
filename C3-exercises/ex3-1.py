from scipy.stats import gamma

k = 3.8         # scale
theta = 400     # shape
x = 200         # probability asked

# CDF
prob = gamma.cdf(x, a=k, scale=theta)
print("Probability of X being less than 200: ")
print(prob)
