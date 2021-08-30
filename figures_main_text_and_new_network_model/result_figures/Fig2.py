import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.ticker as mtick
from matplotlib import lines, patches

import simplejson as json

from rich import print

import bfmplot as bp
from colors import brighter, darker, torgb, tohex

from results import results, a, i30, andx, aval, networks

def get_percental_change(arr,ndx):
    arr = np.array(arr)
    return arr[ndx] / arr[0] *100 - 100

print(results)
c = list(bp.colors)
colors = [
        'black',
        'skyblue',
        'indianred',
        'lightcoral',
        'darkgoldenrod',
        'maroon',
        'darkcyan',
        'mediumvioletred',
        'darkseagreen',
        'crimson',
        'navy',
        ]
c = [ mpl.colors.cnames[_c] for _c in colors ]
c[0] = '#333333'
c[1] = '#666666'
c[2] = '#999999'
#c[2] = brighter

c = [ 
      '#785EF0',
      '#648FFF',
      '#FE6100',
      '#FFB000',
      ]
c[0] = '#333333'
c[1] = '#888888'

#colors = [
#            (0,0,0),       #black
#            (230,159,0),   #orange
#            (86,180,233),  #sky blue
#            (0,158,115),   #blueish gree
#            (240,228,66),  #
#            (0,114,178),
#            (213,94,0),
#            (204,121,167),
#         ]
#
#c = [ tohex(_c) for _c in colors ]
#c[0] = '#333333'
##c[0] = c[-1]
#c[2] = c[5]


DF_main = '4.0'
DF_low = '12.0'
DF_high = '2.4'

# =====================

from results import a
ndx = np.arange(len(a))

c[1] = '#555555'
fig, ax = pl.subplots(1,1,figsize=(5,5),sharey=True)
iax = ax.inset_axes([0.1,0.08,0.3,0.4])
iax.xaxis.set_major_formatter(mtick.PercentFormatter(1,decimals=0))
for inet,net in enumerate(['random','lockdown']):
    DF = DF_main
    if net == 'random':
        marker = 'o'
    else:
        marker = 's'
    for DF in [DF_low, DF_main, DF_high]:
        if DF == DF_main:
            marker = ['o','s'][inet]
        else:
            marker = None
        ax.plot(a,get_percental_change(results[net][0][DF],ndx),c=c[inet],marker=marker,mec='w')
        iax.plot(a,results[net][0][DF],c=c[inet],marker=marker,mec='w',ms=4)

    #ax.plot(a,get_percental_change(results[net][0][DF],ndx),c=c[inet],marker=marker,mec='w')
    #iax.plot(a,results[net][0][DF],c=c[inet],marker=marker,mec='w',ms=4)

    iax.text(0,1.05,r'$\left\langle \Omega \right\rangle/N$',transform=iax.transAxes,ha='center')
    iax.text(1.05,0,r'$a$',transform=iax.transAxes,ha='left')

    iax.set_xlim([0,1])
    iax.set_ylim([0,1])
    ax.set_xlim([0,1])

    ax.xaxis.set_label_position("top")
    ax.xaxis.set_ticks_position("top")
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([-30,0])
#ax.se[inet]t_xticklabels([ labels[net] for net in networks],rotation=40,ha='left',va='bottom')

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100,decimals=0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1,decimals=0))
    #ax[inet].set_xlim(0-dx-lw/2,3+dx+lw/2)
    #ax[inet].set_xlim(-0.5,3.5)
    ax.set_xlabel('app participation $a$      ',loc='right',labelpad=10)

    bp.strip_axis(iax)

bp.tools.set_n_ticks(ax,nx=5,ny=4)
ax.set_ylabel('outbreak size reduction',loc='bottom')
fig.tight_layout()
pl.subplots_adjust(wspace=0.15)

# add curve labels

#for inet, net in enumerate(['random','lockdown']):
#    for DF,cut,yoff,bg,ha in zip([ DF_low, DF_main, DF_high],
#                              [ -2,-2, len(DF_high) ],
#                              [0.04,0.03,-0.08],
#                              ['w','#eeeeee','w'],
#                              ['left','left','right'],
#                             ):
#        bp.add_curve_label(ax[inet],
#                           a,
#                           get_percental_change(results[net][0][DF],ndx),
#                           '  DF$_0$ = '+DF[:cut]+"  ",
#                           label_pos_rel=0.75,
#                           bbox_facecolor='None',
#                           y_offset=yoff,
#                           x_offset=0.01,
#                           angle = 0,
#                           ha=ha,
#                           )


fig.savefig('lockdown_2.pdf',dpi=300)

pl.show()
