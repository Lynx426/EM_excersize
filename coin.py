#created by Tekrammeena
#Roll no. B21CI042
#IIT JODHPUR
#03-07-2023

#IMPORT NUMPY LIBRARY
import numpy as np

def coin_em(rolls, theta_A=None, theta_B=None, maxiter=10):
    # Initial Guess
    theta_A = theta_A or random.random()  #initialize it with a random value
    theta_B = theta_B or random.random()  # initialize it with a random value
    thetas = [(theta_A, theta_B)]  # Store the initial values of theta_A and theta_B in a list
    # Iterate
    for i in range(maxiter): # Perform the Expectation-Maximization algorithm
        print("#%d:\t%0.2f %0.2f" % (i, theta_A, theta_B))  # Print the current iteration number and the values
        heads_A, tails_A, heads_B, tails_B = e_step(rolls, theta_A, theta_B)  # Perform the E-step 
        theta_A, theta_B = m_step(heads_A, tails_A, heads_B, tails_B)  # Perform the M-step 
        
    thetas.append((theta_A, theta_B))  # Append the final values of theta_A and theta_B to the list    
    return thetas, (theta_A, theta_B)


def e_step(rolls, theta_A, theta_B):
    
    # Initialize the counts
    heads_A, tails_A = 0, 0
    heads_B, tails_B = 0, 0

    # Loop through each trial in the list of rolls
    for trial in rolls:
        # Calculate the likelihood 
        likelihood_A = coin_likelihood(trial, theta_A)
        likelihood_B = coin_likelihood(trial, theta_B)

        # Calculate the posterior probability 
        p_A = likelihood_A / (likelihood_A + likelihood_B)
        p_B = likelihood_B / (likelihood_A + likelihood_B)

        # Accumulate the expected number 
        heads_A += p_A * trial.count("H")
        tails_A += p_A * trial.count("T")
        heads_B += p_B * trial.count("H")
        tails_B += p_B * trial.count("T") 

    # Return the expected counts 
    return heads_A, tails_A, heads_B, tails_B

def m_step(heads_A, tails_A, heads_B, tails_B):
    """Produce the values for theta that maximize the expected number of heads/tails"""

     # Calculate the new theta values using the accumulated expected counts
    theta_A = heads_A / (heads_A + tails_A) 
    theta_B = heads_B / (heads_B + tails_B)
    return theta_A, theta_B


def coin_likelihood(roll, bias):
    # P(X | Z, theta)
    numHeads = roll.count("H")  # Count the number of "H" (heads) in the roll
    flips = len(roll)  # Count the total number of coin flips in the roll
    return pow(bias, numHeads) * pow(1-bias, flips-numHeads)  # Calculate the likelihood of the roll 


x = ['HHHHHTTTTTHHH', 'HHTHHTHTHTHHT', 'HHTHTTHTHTHTT', 'TTTHTTHTHTHTT', 'THTHTTHTHTHHH']
thetas, _ = coin_em(rolls = x, theta_A = 0.6, theta_B = 0.5, maxiter=6)



# %matplotlib inline
from matplotlib import pyplot as plt
import matplotlib as mpl

# Function to plot the likelihood 
def plot_coin_likelihood(rolls, thetas=None, figout='Documents/base.png'):
    # Generate a grid of theta_A and theta_B values
    xvals = np.linspace(0.01, 0.99, 100)
    yvals = np.linspace(0.01, 0.99, 100)
    X, Y = np.meshgrid(xvals, yvals)
    
    # Compute likelihood for each pair of theta_A and theta_B
    Z = []
    for i, r in enumerate(X):
        z = []
        for j, c in enumerate(r):
            z.append(coin_marginal_likelihood(rolls, c, Y[i][j]))
        Z.append(z)
    
    # Plot the likelihood using contour plots
    plt.figure(figsize=(10, 8))
    C = plt.contour(X, Y, Z, 150)
    cbar = plt.colorbar(C)
    plt.title(r"Likelihood $\log p(\mathcal{X}|\theta_A,\theta_B)$", fontsize=20)
    plt.xlabel(r"$\theta_A$", fontsize=20)
    plt.ylabel(r"$\theta_B$", fontsize=20)

    # Plot thetas if provided
    if thetas is not None:
        thetas = np.array(thetas)
        # Plot lines connecting thetas
        plt.plot(thetas[:, 0], thetas[:, 1], '-k', lw=2.0)
        # Plot markers for thetas
        plt.plot(thetas[:, 0], thetas[:, 1], 'ok', ms=5.0)

# Function to calculate the marginal likelihood
def coin_marginal_likelihood(rolls, biasA, biasB):
    trials = []
    for roll in rolls:
        h = roll.count("H")  # Count the number of heads in the roll
        t = roll.count("T")  # Count the number of tails in the roll
        likelihoodA = coin_likelihood(roll, biasA)  # Likelihood of the roll
        likelihoodB = coin_likelihood(roll, biasB)  # Likelihood of the roll
        trials.append(np.log(0.5 * (likelihoodA + likelihoodB)))  # Compute log marginal likelihood 
    return sum(trials)  # Return the sum of log marginal likelihoods 

plot_coin_likelihood(x, thetas)  # plot_coin_likelihood(X, thetas)
plt.savefig("./Image/plot_coin_likelihood.png")
print(x)
