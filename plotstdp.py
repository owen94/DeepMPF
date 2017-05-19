import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# a = [-102, -3, 100, 5, 6, 2, 7, -2, 10, 5,  5.8,5.9, 5.7, 6.9,3,5.5, 6.5,
#      5, 16, 17, 29, 28, 39, 78, 88, 60, -6, -8.3, -8.3, -6.3, -6.5, -6.2, -18, -22, -19,
#      -23,-22, -19, -67, -65, -68, -77, -80, -42]
#
# b = [-1.6/100, -42/100, 3/100, 100/100, .93, .90, .86, .74, .70, .52,.49, .42, .30, .30, .26, .19,
#      .17, .02, .30, .40, .13, .05, .2, .11, -0.04, -0.1, -.32,-.25,-.21,-.18, -.12, 0, -.3, -.23, -.18,
#      -.08, -.14, -0.03, -.08, -.03, 0, .02, -.12, -.15]
#
#
# plt.plot(a,b,'ro', mfc='none' )
# yticks = [-0.6, -0.4, -0.2, 0,0.2, 0.4,0.6, 0.8, 1.0, 1.2]
# plt.yticks(yticks,fontsize = 14)
# xticks = [-80, -40, -0, 0,40, 80]
# plt.yticks(yticks,fontsize = 10)
# plt.xticks(xticks,fontsize = 10)
# plt.axis([-120, 120, -0.6, 1.2])
# plt.plot([-120, 120], [0, 0], '--')
# plt.plot([0, 0], [-120, 120], '--')
#
# plt.xlabel('Spikes timing (ms)', fontsize = 12)
# plt.ylabel('Weight updates $\\frac{ \\Delta W_{ij}}{W_{ij}}$', fontsize = 12)
#
#
#
# #plt.show()
# plt.savefig( '../DBN_results/STDP.eps', bbox_inches='tight')


x = np.arange(-0.6, -0.013, 0.001)
y = 2/x
x1 = np.arange(0.008, 0.6, 0.001)
y1 = 2/x1

plt.plot(x,y,'r')
plt.plot(x1,y1, 'r')
plt.plot([-10, 10], [0, 0], '--')
plt.plot([0, 0], [-60, 120], '--')

plt.xlabel('Expected time to post-synaptic spike $1/\\delta_{j}$', fontsize = 12)
plt.ylabel('Weight updates $\\Delta W_{ij} $', fontsize = 12)
plt.axis([-0.6, 0.6, -60, 120])
plt.xticks( color='w', fontsize = 10)
plt.yticks( color= 'w', fontsize = 3)

# fig = plt.gcf()
# fig.subplots_adjust(bottom=0.2)
#plt.show()
plt.savefig( '../DBN_results/weightsupdate.eps', bbox_inches='tight')











