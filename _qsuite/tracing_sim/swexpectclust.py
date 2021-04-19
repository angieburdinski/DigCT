#from smallworld.theory import get_effective_medium_eigenvalue_gap
from smallworld.theory import get_effective_medium_eigenvalue_gap
from smallworld.theory import expected_clustering
from bfmplot import pl as pl
import bfmplot as bp
import numpy as np
from smallworld import get_smallworld_graph
#N = 200_001
N = 1_001
betas = np.logspace(-10,0,50, endpoint = False)
k_over_2 = 10
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
