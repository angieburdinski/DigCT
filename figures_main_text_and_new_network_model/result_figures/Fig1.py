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
#c[1] = '#888888'
c[1] = '#5540BF'

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

fig, ax = pl.subplots(1,1,figsize=(4,4.5))

DF_main = '4.0'
DF_low = '12.0'
DF_high = '2.4'

def get_percental_change(arr,ndx):
    arr = np.array(arr)
    return arr[ndx] / arr[0] *100 - 100

scl = 0.9
w = scl*0.5
lw = scl*w/1.5
dx = scl*(w/2 + lw/10)

labels = {
        'random' : 'random structure\nhomogen. contacts',
        'exp' : 'random structure\nexponential',
        'exp_sw' : 'locally clustered\nsmall-world',
        'sw' : 'locally clustered\nsmall-world',
        }

networks = ['random','sw','exp','exp_sw']
vals = {net: {} for net in networks }

for inet, net in enumerate(networks):
    vals[net][DF_low] = get_percental_change(results[net][0][DF_low],i30)
    vals[net][DF_high] = get_percental_change(results[net][0][DF_high],i30)
    vals[net][DF_main] = get_percental_change(results[net][0][DF_main],i30)
    basecolor = c[inet]
    edgecolor = brighter(torgb(basecolor),1/3)
    ax.bar([inet+dx],[vals[net][DF_high]],color=brighter(torgb(c[inet]),3),width=lw,edgecolor=basecolor)
    ax.bar([inet],[vals[net][DF_main]],color=c[inet],width=w,edgecolor=basecolor)
    ax.bar([inet-dx],[vals[net][DF_low]],color=brighter(torgb(c[inet]),3),width=lw,edgecolor=basecolor)
    ax.xaxis.set_label_position("top")
    ax.xaxis.set_ticks_position("top")
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim([-20,0])
ax.set_xticks(range(len(networks)))
#ax.set_xticklabels([ labels[net] for net in networks],rotation=40,ha='left',va='bottom')
ax.set_xticklabels(['' for net in networks])

ax.yaxis.set_major_formatter(mtick.PercentFormatter(100,decimals=0))
bp.tools.set_n_ticks(ax,nx=None,ny=5)
ax.set_ylabel('outbreak size reduction',loc='bottom')
ax.set_xlim(0-dx-lw/2,3+dx+lw/2)
ax.set_xlim(-0.5,3.5)

ax.text(1.0,0.0,'app participa-\ntion $a$ = 30%',transform=ax.transAxes,va='bottom',ha='right')

#disregard 80%
#andx = andx[:-1]

inet = 3
net = networks[inet]

#ax.annotate('DF = '+DF_main,
#             xy=[
#                    inet - dx -lw/2,
#                    vals[net][DF_low],
#               ],
#             xytext=[
#                    inet + 1,
#                    vals[net][DF_low],
#               ],
#             xycoords='data',
#             va = 'center',
#             arrowprops={
#                 #'width':0.5,
#                 'arrowstyle':'-',
#
#             #       'connectionstyle':'-',
#                 }
#               )
ax.axhline(-9, color = 'k', alpha = 0.5, ls = 'dotted')
for iDF, DF in enumerate([DF_low,DF_main,DF_high]):
    lbl = DF
    if lbl[-1] == '0':
        lbl = lbl[:-2]
    ax.text(inet + 0.55, vals[net][DF],
            'DF${}_0$ = '+lbl,
            va='center',
            ha='left',
            fontsize='medium' if iDF == 1 else 'small',
            color ='#333333',
            transform=ax.transData,
            )
ax.plot(
             [
                    inet - dx -lw/2,
                    inet + 1,
             ],
             [
                    vals[net][DF_low],
                    vals[net][DF_low],
             ],
             c='#333333',
             ls='--',
             lw = 0.8,
         )
ax.plot(
             [
                    inet - w/2,
                    inet + 1,
             ],
             [
                    vals[net][DF_main],
                    vals[net][DF_main],
             ],
             c='#333333',
             ls='-',
             lw = 0.8,
        )

ax.plot(
             [
                    inet + dx - lw/2,
                    inet + 1,
             ],
             [
                    vals[net][DF_high],
                    vals[net][DF_high],
             ],
             c='#333333',
             ls='--',
             lw = 0.8,
        )


line = lines.Line2D([0,0,2,2],[0,5,5,0],color='k',lw=1)
line.set_clip_on(False)
ax.add_line(line)
line = lines.Line2D([1,1,3,3],[0,3,3,0],color='k',lw=1)
line.set_clip_on(False)
ax.add_line(line)

ax.text(1,5,'random\nstructure',va='center',ha='center',bbox=dict(facecolor='w',edgecolor='None'))
ax.text(2.5,3,'locally clustered\nsmall-world',va='center',ha='left',bbox=dict(facecolor='w',edgecolor='None'))

#rect1 = patches.Rectangle([-0.5,20],width=3,height=5,facecolor='#333333',edgecolor='None')
#rect1.set_clip_on(False)
#ax.add_patch(rect1)

#fig.tight_layout(rect=[0, 0.03, 1, 2])
fig.tight_layout()

fig.savefig('outbreak_reduction.pdf',dpi=300)


#========================
andx = andx[:-1]
aval = aval[:-1]

fig, axs = pl.subplots(2,2,figsize=(6,6),sharex=True,sharey=False)
for a in axs.flatten():
    bp.strip_axis(a)

for inet, (ax, net) in enumerate(zip(axs.flatten(),networks)):
   # ax.text(0-lw,results[net][0]['inf'][0]+0.03,'no testing',ha='left',va='bottom',rotation=60,transform=ax.transData)
    if inet == 1:
        ax.text(0-lw,results[net][0]['inf'][0]+0.03,'no testing',ha='center',va='bottom',rotation=90,transform=ax.transData,color='k')
        ax.text(0-lw/3+lw/4,results[net][0][DF_main][0]+0.02,'DF${}_0$ = '+DF_main[:-2],ha='left',va='bottom',rotation=90,transform=ax.transData,color='k')
        ax.text(0+lw/3+2*lw/4,results[net][0][DF_high][0]+0.02,'DF${}_0$ = '+DF_high,ha='left',va='bottom',rotation=90,transform=ax.transData,color='#666666')
    #ax.text(0-lw/3,results[net][0][DF_main][0]+0.03,'DF${}_0$ = '+DF_main,ha='left',va='bottom',rotation=60,transform=ax.transData)
    #ax.text(0+lw/3,results[net][0][DF_high][0]+0.03,'DF${}_0$ = '+DF_low,ha='left',va='bottom',rotation=60,transform=ax.transData)
    ax.text(1-lw/3,results[net][0][DF_main][i30]+0.03,'current',ha='center',va='bottom',rotation=90,transform=ax.transData,color='#333333')
    ax.bar([0-lw],[results[net][0]['inf'][0]],width=lw,color='k',edgecolor='k',linewidth=0.5)
    ax.bar([0-lw/3-0.012],[0.004+results[net][0][DF_main][0]],width=lw,color='w',edgecolor='w',linewidth=0.5)

    for DF,i in zip([DF_main, DF_high],[-1,1]):
        arr = np.array(results[net][0][DF])[andx]
        basecolor = c[inet]
        if i == 1:
            color = brighter(torgb(basecolor),3)
            edgecolor = brighter(torgb(basecolor),1/3)
        else:
            color = basecolor
            edgecolor = brighter(torgb(basecolor),1/3)

        x = np.arange(len(andx))
        ax.bar(x+i*lw/3,arr,width=lw,color=color,edgecolor=edgecolor,linewidth=0.5)

    ax.set_ylim([0,1.0])

    ax.annotate('',
            xy=(1+lw/3+lw/3, results[net][0][DF_high][i30]-0.01), xycoords='data',
            xytext=(1-lw/3+lw/2, results[net][0][DF_main][i30]), textcoords='data',
            arrowprops=dict(arrowstyle="-|>",
                            facecolor='k',
                            connectionstyle="angle,angleA=0,angleB=90,rad=2"))
    ax.annotate('',
            xy=(2-lw/3+lw/3, results[net][0][DF_main][andx[2]]-0.01), xycoords='data',
            xytext=(1-lw/3+lw/2, results[net][0][DF_main][i30]), textcoords='data',
            arrowprops=dict(arrowstyle="-|>",
                            facecolor='k',
                            connectionstyle="angle,angleA=0,angleB=90,rad=2"))


    #ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0,decimals=0))
    #ax.set_yticks([0,0.25,0.5,0.75,1.0])
    #ax.set_yticklabels(['0','N/4','N/2','3N/4','N'])

    if inet % 2 == 0:
        ax.set_ylabel(r'outbreak size $\left\langle \Omega\right\rangle/N$',loc='top')
    else:
        ax.set_yticklabels(['' for t in ax.get_yticks()])

#axs[0,0].text(1.33,0.05,'incr. testing',rotation=90)
#axs[0,0].text(1.35,0.48,'increased\nparticipation',rotation=0,)
axs[0,0].text(1.32,0.35,'incr. testing',rotation=90)
axs[0,0].text(1.35,0.82,'increased\nparticipation',rotation=0,)
#axs[0,1].set_yticklabels(['','','',''])
fig.tight_layout()
axs[0,0].set_xticks(range(len(aval)))
axs[0,0].set_xticklabels(['{0:d}%'.format(int(v*100)) for v in aval])
#axs[1,0].text(1.1,-0.2,'app participation $a$',ha='center',transform=axs[1,0].transAxes)
for i in range(2):
    axs[1,i].set_xlabel('app participation $a$      ',loc='right')

fig.tight_layout()
pl.subplots_adjust(wspace=0.1)
print(aval)

fig.savefig('structure_comparison.pdf')

# =====================

from results import a
ndx = np.arange(len(a))

c[1] = '#555555'

fig, ax = pl.subplots(1,2,figsize=(10,3.5),sharey=True)

for inet,net in enumerate(['random','lockdown']):
    iax = ax[inet].inset_axes([0.1,0.08,0.3,0.4])
    iax.xaxis.set_major_formatter(mtick.PercentFormatter(1,decimals=0))
    for DF in [DF_low, DF_main, DF_high]:
        if DF == DF_main:
            marker = 'o'
        else:
            marker = None
        ax[inet].plot(a,get_percental_change(results[net][0][DF],ndx),c=c[inet],marker=marker,mec='w')
        iax.plot(a,results[net][0][DF],c=c[inet],marker=marker,mec='w',ms=4)

    ax[inet].fill_between(a,
                           get_percental_change(results[net][0][DF_low],ndx),
                           get_percental_change(results[net][0][DF_high],ndx),
                           color='#eeeeee',)
    ax[inet].fill_between(a,
                           get_percental_change(results[net][0][DF_low],ndx),
                           get_percental_change(results[net][0][DF_high],ndx),
                           color='#eeeeee',)
    iax.fill_between(a,
                           (results[net][0][DF_low]),
                           (results[net][0][DF_high]),
                           color='#eeeeee',)
    iax.fill_between(a,
                           (results[net][0][DF_low]),
                           (results[net][0][DF_high]),
                           color='#eeeeee',)
    iax.text(0,1.05,r'$\left\langle \Omega \right\rangle/N$',transform=iax.transAxes,ha='center')
    iax.text(1.05,0,r'$a$',transform=iax.transAxes,ha='left')

    iax.set_xlim([0,1])
    ax[inet].set_xlim([0,1])

    ax[inet].xaxis.set_label_position("top")
    ax[inet].xaxis.set_ticks_position("top")
    ax[inet].spines['bottom'].set_visible(False)
    ax[inet].spines['right'].set_visible(False)
    ax[inet].set_ylim([-50,0])
#ax.se[inet]t_xticklabels([ labels[net] for net in networks],rotation=40,ha='left',va='bottom')

    ax[inet].yaxis.set_major_formatter(mtick.PercentFormatter(100,decimals=0))
    ax[inet].xaxis.set_major_formatter(mtick.PercentFormatter(1,decimals=0))
    bp.tools.set_n_ticks(ax[inet],nx=5,ny=4)
    #ax[inet].set_xlim(0-dx-lw/2,3+dx+lw/2)
    #ax[inet].set_xlim(-0.5,3.5)
    ax[inet].set_xlabel('app participation $a$      ',loc='right',labelpad=10)

    bp.strip_axis(iax)

ax[0].set_ylabel('outbreak size reduction',loc='bottom')
fig.tight_layout()
pl.subplots_adjust(wspace=0.2)

# add curve labels

for inet, net in enumerate(['random','lockdown']):
    for DF,cut,yoff,bg,ha in zip([ DF_low, DF_main, DF_high],
                              [ -2,-2, len(DF_high) ],
                              [0.04,0.03,-0.08],
                              ['w','#eeeeee','w'],
                              ['left','left','right'],
                             ):
        bp.add_curve_label(ax[inet],
                           a,
                           get_percental_change(results[net][0][DF],ndx),
                           '  DF$_0$ = '+DF[:cut]+"  ",
                           label_pos_rel=0.75,
                           bbox_facecolor='None',
                           y_offset=yoff,
                           x_offset=0.01,
                           angle = 0,
                           ha=ha,
                           )


fig.savefig('lockdown.pdf',dpi=300)

pl.show()
