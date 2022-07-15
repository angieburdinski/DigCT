from smallworld.theory import get_effective_medium_eigenvalue_gap
from smallworld.theory import expected_clustering
import matplotlib.pyplot as pl
import numpy as np

def get_beta(N,k_over_2):
    """
    This function is used to get the long range redistribution parameter $\beta$
    to build locally clustered networks with a localized degree distribution
    (small-world).
    Parameter
    -----------
    N : number of nodes (int) must be odd
    k_over_2 : degree of nodes / 2 (int)

    Return
    -------
    Plot with clustering coefficient and mixing time for different long range
    redistribution parameter.
    """
    betas = np.logspace(-10,0,50, endpoint = False)
    x = [expected_clustering(N,k_over_2,beta) for beta in betas]
    y = [1/get_effective_medium_eigenvalue_gap(N,k_over_2,beta) for beta in betas]
    fig, ax1 = pl.subplots()
    ax1.set_xlabel(r'long range redistribution parameter $\beta$')
    ax1.set_ylabel('clustering coefficient',color='#666666')
    ax1.plot(betas, x, color='#666666')
    ax1.tick_params(axis='y', labelcolor='#666666')
    pl.xscale('log')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('mixing time',color='#1b9e77')  # we already handled the x-label with ax1
    ax2.plot(betas, y,color='#1b9e77')
    ax2.tick_params(axis='y', labelcolor='#1b9e77')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    pl.show()
if __name__ == "__main__":
    N = 10001
    k_over_2 = 10
    get_beta(N,k_over_2)
