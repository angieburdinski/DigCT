import numpy as np
import matplotlib.pyplot as plt
df  = np.load('_qsuite/tracing_sim/results_smallworld_exponential_asc_withQ_NMEAS_100_ONLYSAVETIME_False/results_mean_err.npz')
df = df['mean']
O = np.array([sum([df[:,x,0,i] for i in range(5)]) for x in range(4)])/200_000
darkfactor = (O)/(np.array([sum([df[:,x,1,i] for i in [2,3]]) for x in range(4)])/200_000)
for i in range(4):
    plt.plot(darkfactor[i])
xpositions = (0, 6, 12, 18, 24)
xlabels = ('0%',"25%","50%", "75%",'100%')
plt.xticks(xpositions,xlabels)
plt.ylabel("DF")
plt.xlabel("app participation")
plt.show()
