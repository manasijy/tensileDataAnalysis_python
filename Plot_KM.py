# This program is used for line plots. Data is read from matfile or csv or xsls 

import scipy.io
import numpy as np
#import pandas as pd #needed to read csv files
import h5py #to read matfile data
import matplotlib.pyplot as plt

# Read data from matfile 'testData.mat'

f1= h5py.File(r'C:\Users\npg\OneDrive\Post_aging_of_FSW\Aged_FSP_DIC\as_fsp\421_fsp\Specimen_RawData_1testData.mat','r')#_AsFSP
f2= h5py.File(r'C:\Users\npg\OneDrive\Post_aging_of_FSW\Aged_FSP_DIC\fsp_T6\421_24hr_T6\Specimen_RawData_1testData.mat','r')#_T6
f3= h5py.File(r'C:\Users\npg\OneDrive\Post_aging_of_FSW\Aged_FSP_DIC\fsp_T8\421_24hr_prest2\Specimen_RawData_1testData.mat','r')#_2T8
f4= h5py.File(r'C:\Users\npg\OneDrive\Post_aging_of_FSW\Aged_FSP_DIC\fsp_T8\421_24hr_prest4\Specimen_RawData_1testData.mat','r')#_4T8
f = [f1, f2, f3, f4]
#to open the matfile - handle f
#r used to handle py bug - to indicate it is raw string
#To get specific variables
fig = plt.figure()
ax = plt.axes()
c = ['r', 'b', 'g',  'm']
for i in range(len(f)):

    data1 =f[i].get('KMX')
    data2= f[i].get('KMY')
    KMX = np.array(data1)
    KMY = np.array(data2)
    j=np.size(KMY)-100
    ax.plot(KMX[0,0:j]-KMX[0,0], KMY[0,0:j],c[i])

ax.set_xlabel('$\\sigma$ - $\\sigma_{y}$ (MPa)',fontsize=14)
ax.set_ylabel('d$\\sigma$ / d$\\epsilon$ (MPa)',fontsize=14)
ax.set(xlim = [0, 150],ylim = [0, 12000]) #title = ' True Stress vs., Plastic Strain',    
ax.legend(['As FSP', 'FSP+24hrs@$155^o$C', 'FSP+2%prestretch+24hrs@$155^o$C', 'FSP+4%prestretch+24hrs@$155^o$C'],loc='upper right',fontsize=12)
ax.tick_params(axis='both', which='both', direction='in',labelsize=14,top=True, right=True)
ax.set_xticks([0, 50, 100, 150])
ax.set_xticklabels(['0', '50', '100', '150'])
ax.set_yticks([0, 4000, 8000, 12000])
ax.set_yticklabels(['0', '4000', '8000', '12000'])

plt.show()
f1.close()
f2.close()
f3.close()
f4.close()

#print(EStr[0,1:100])
#print(EStn[0,100:150])

##################################################################
##################################################################

## Read data from csvfile 'Filename.csv'
#csvFile = pandas.read_csv('Filename.csv', usecols = ['IQ','Scores'])
#print(csvFile)




##################################################################
##################################################################
#Plot Styles setup

##plt.style.use('classic')
##fig=plt.figure(figsize=(5,3))
##ax = plt.axes(facecolor='#E6E6E6')
### Display ticks underneath the axis
##ax.set_axisbelow(True)
### White frame
##plt.grid(color='w', linestyle='solid')
##
### Hide the frame
##for spine in ax.spines.values():    
##    spine.set_visible(False)
##    
### Hide the markers at the top and the right
##ax.xaxis.tick_bottom()
##ax.yaxis.tick_left()
##
### We can personalise the markers, and rotate them
##marqueurs = [-3, -2, -1, 0, 1, 2, 3]
##xtick_labels = ['A', 'B', 'C', 'D', 'E', 'F']
##plt.xticks(marqueurs, xtick_labels, rotation=30)
##
### Change the color of markers
##ax.tick_params(colors='gray', direction='out')
##for tick in ax.get_xticklabels():    
##    tick.set_color('gray')
##for tick in ax.get_yticklabels():    
##    tick.set_color('gray')    
##    
### Change the color of the edges
##ax.hist(x, edgecolor='#E6E6E6', color='#EE6666');

##################################################################
##################################################################


#plt.rc('lines', linewidth=2, color='r')
#plt.rc('axes', facecolor='r')

#multiple subplots
##    fig, axs = plt.subplots(2)
##    fig.suptitle('Vertically stacked subplots')
##    axs[0].plot(x, y)
##    axs[1].plot(x, -y)
# >>> fig, _ = plt.subplots() # >>> type(fig) #: _ used to recieve a throwaway variable
# type is used to get the variable type
# heirarchy eg : one_tick = fig.axes[0].yaxis.get_major_ticks()[0]

# fig, ax = plt.subplots(figsize=(5, 3))# creates one figure with one plot axes
# fig.tight_layout() #- to remove white space in the figure
# ax.set_xlim(xmin=yrs[0], xmax=yrs[-1]) # setting limits on axes range
#fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
#...                                figsize=(8, 4))# creates 2 subplots
#ax2.hist(data, bins=np.arange(data.min(), data.max()),
#...          label=('x', 'y'))# creates histogram
#subplot2grid() # creates different size subplots

#
##gridsize = (3, 2)
##fig = plt.figure(figsize=(12, 8))
##ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
##ax2 = plt.subplot2grid(gridsize, (2, 0))
##ax3 = plt.subplot2grid(gridsize, (2, 1))

##sctr = ax1.scatter(x=age, y=pop, c=y, cmap='RdYlGn')
##plt.colorbar(sctr, ax=ax1, format='$%d')
##ax1.set_yscale('log')
##ax2.hist(age, bins='auto')
##ax3.hist(pop, bins='auto', log=True)

#plt.close('all') # plt.close(num) closes the figure number num

##We can also create sub plots
##fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
##ax.shape
##ax1, ax2, ax3, ax4 = ax.flatten()  # flatten a 2d NumPy array to 1d
# $x$ - itallic  x







###color='0.75', linestyle='dashdot', label='gris')
### color='#FF0000', linestyle='dotted', label='rouge')
### Axis limits. Try also 'tight' and 'equal' to see their effect
##plt.title("Example of a graph")
##plt.legend(loc='lower left');
##ax = ax.set(xlabel='x', ylabel='sin(x)')




    

