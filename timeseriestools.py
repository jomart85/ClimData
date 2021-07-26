'''
This set of functions is essentially used for manipulating a time series of climate data. It doesn't provide any sort of actual
statistical mainpulations of said data, but rather prepares it for future use. Blahh blah blah blah

'''

import numpy as np
import scipy.stats as spf
import scipy.fftpack as spf
import xarray as xr
import ertools as er
import regiontools as rt
import statstools as st
#====================================================================================
#Function that getting the running mean of a time series
def runningAverage(vector, k):

    #initializing the smoothed vector as an empty array
    smoothedVector = np.array([])
    totalPoints = 2*k+1
    for i in range(k,len(vector)-k):
        span = range(i-k,i+k)
        mean = np.sum(vector[(i-k):(i+k+1)])/totalPoints
        smoothedVector = np.append(smoothedVector,mean)

    return smoothedVector
        
#====================================================================================
#Function that puts a data set into an annual scale
def annualScale(sst):
    years = int(len(sst)/12)
    months = 12

    sstAnnual = np.copy(sst)

    sstAnnual.shape = (years,months)
    sstAnnual = np.mean(sstAnnual,axis=1)

    return sstAnnual
#====================================================================================
#Gathering the climatology of a given time series
def climatology(sst):
    yrs = int(len(sst)/12)

    clim = np.copy(sst)

    clim.shape = (yrs,12)

    clim = np.mean(clim, axis=0)

    return clim
        

#====================================================================================
#Separate function for the removal of the seasonal cycle
def removeCycle(sst):
    anoms = np.copy(sst)
    useableYears = int(np.floor(len(sst)/12))

    indices = np.arange(0,len(sst)) % 12
    climo = np.copy(anoms[:(useableYears * 12)])

    climo.shape = (useableYears, 12)
    climo = np.mean(climo, axis=0)

    for m in range(12):
        k = indices == m
        anoms[k] = anoms[k] - climo[m]

    return anoms

#====================================================================================
#Now, for a function that adds a cycle into the time series for some reason
def addCycle(x,cycle):
    y = np.copy(x)
    years = int(len(y)/12)
    y.shape = (years,12)
    for j in range(12):
        y[:,j] = y[:,j] + cycle[j]

    return y

#======================================================================================================
#Something for the maximum AMOC index
def maxAMOC(AMOC, lat, z, atLat=26.5):
    dist = abs(lat - atLat)
    dist = dist == np.min(dist)

    maxAMOC = AMOC[:,z>500,dist]
    if(np.sum(dist)>1):
        maxAMOC = np.mean(maxAMOC,axis=2)

    maxAMOC = np.max(maxAMOC,axis=1)
    maxAMOC = np.squeeze(maxAMOC)
    return maxAMOC

#======================================================================================================
#Function for the decorrelation time scale of a time series. This is based off of fitting the data
#to an AR-1 process
def decorrtime(x):
    t = np.array([1.])
    k = -1
    j = 1
    ntot = len(x)
    phi = st.fitAR1(x,1)
    while (j<(ntot-1)):
        ti = 2 * phi**(2*j)# (st.fitAR1(x,j) ** 2)
        #print(j,': ',ti)
        t = np.append(t, ti)#(st.standardCorr(x[j:],x[:k]) ** 2)
        j += 1
    t = np.nansum(t)
    return t

#=======================================================================================================
#Function that selects the season from the data set
def selectSeason(x,season='winter'):
    season_dict_xr = {'winter':'DJF','spring':'MAM','summer':'JJA','fall':'SON'}
    season_dict_np = {'winter':[11,0,1],'spring':[2,3,4],'summer':[5,6,7],'fall':[8,9,10]}
    y = []
    if(type(x) is np.ndarray):
        time = len(x)
        nyrs = int(time/12)
        if (nyrs*12 == time):
            z = np.copy(x)
            z.shape = (nyrs,12)
            y = np.zeros(nyrs)
            seas = season_dict_np[season]
            for s in seas:
                y += z[:,s]

            y = y/3

        else:
            print('invalid length of time series. has to be divisible by 12')

    elif(type(x) is xr.Dataset):
        print('Under construction')

    return y