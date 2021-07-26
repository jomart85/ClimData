'''
These functions can perform basic analysis (detrending, climatology removal, etc) on a gridded data set
'''
import numpy as np
import numpy.linalg as nlg
import haditools as ht
import ertools as reader
import matrixtools as mt
import statstools as st
import timeseriestools as ts
import scipy.interpolate as spi
import scipy.stats as sps
import xarray as xr

#===========================================================================================
#A function used for smoothing out a gridded data set
#Input:
#   S(time,lat,lon): The gridded data set to be smoothed
#   k: The extent both prior and after a given point over whichh to smooth the data
#      (the full smoothing window will actually be 2k+1)
#Output:
#   R(time-2*k,lat,lon): The gridded data set thhat has been smoothed
def runningAverageRegion(S,k):
    (time,rows,cols) = S.shape

    R = np.zeros([time - 2*k,rows,cols])
    j = 0
    for i in range(k,time - k):
        R[j,:,:] = np.sum(S[(i-k):(i+k+1),:,:],axis=0)/(2*k + 1)
        j += 1
    return R

#===========================================================================================
#A function that gathers the AR1 coefficients from Gilman (1963) for each grid point of a given
#data set
def gridAR1(X,lag):
    (time,rows,cols) = X.shape
    RHO = np.zeros((rows,cols))

    for i in range(rows):
        for j in range(cols):
            x_ij = np.copy(X[:,i,j])
            if(np.isnan(x_ij).any()):
                RHO[i,j] = np.nan
            else:
                RHO[i,j] = st.fitAR1(x_ij,lag)

    return RHO    

#===========================================================================================
#Function that returns the annual time scale of gridded data
#Assumes that the grid is in a (time, space) format

def annualScaleRegion(X):

    if(type(X) is np.ndarray):
        #Will take two cases: a (time, space) format and a (time,rows,cols) format
        months = X.shape[0]
        years = int(months/12)
        #Creating the new matrix for the annual SST values
        S = np.copy(X)
        dim = len(S.shape)
        if (dim==2):
            (time,space) = S.shape
            S.shape = (years,12,space)
        elif(dim==3):
            (time,rows,cols) = S.shape
            S.shape = (years,12,rows,cols)

        S = np.mean(S,axis=1)

    elif(type(X) is xr.Dataset ):
        S = X.groupby('time.year').mean('time')

    else:
        print('Invalid Data type')
        S = []

    return S

#===============================================================================================
#Developing a correlation map for a grid and a time series
def corrGrid(u,X):
    (nt,nx,ny) = X.shape
    corrs = np.zeros((nx,ny))

    for i in range(nx):
        for j in range(ny):
            if(np.isnan(X[0,i,j])):
                corrs[i,j] = np.nan
            else:
                corrs[i,j] = np.corrcoef(u,X[:,i,j])[0,1]

    return corrs
#===============================================================================================
#Getting the correlation map between a grid and a time series that includes 95% P-values
def pearsonGrid(u,X):

    (time,rows,cols) = X.shape

    corMap = np.zeros((rows,cols))
    pvalMap = np.zeros((rows,cols))

    for i in range(rows):
        for j in range(cols):
            if(np.isnan(X[0,i,j])):
                corMap[i,j] = np.nan
                pvalMap[i,j] = np.nan
            else:

                ptest = sps.pearsonr(u,X[:,i,j])
                corMap[i,j] = ptest[0]
                pvalMap[i,j] = ptest[1]

    return corMap, pvalMap

#===============================================================================================
#Getting the correlation map between two grids. 
# Note: both grids have to have the same dimensions
def pearson2Grids(X,Y):

    (rows,cols) = X.shape[1:]

    corMap = np.zeros((rows,cols))
    pvalMap = np.zeros((rows,cols))

    for i in range(rows):
        for j in range(cols):
            if(np.isnan(X[0,i,j]) or np.isnan(Y[0,i,j])):
                corMap[i,j] = np.nan
                pvalMap[i,j] = np.nan
            else:

                ptest = sps.pearsonr(X[:,i,j],Y[:,i,j])
                corMap[i,j] = ptest[0]
                pvalMap[i,j] = ptest[1]

    return corMap, pvalMap

#===============================================================================================
#Regressing gridded data onto a time series
def regressGrid(X,y):
    (rows,cols) = X.shape[1:]
    M = np.zeros((rows,cols))

    y_is_series = (len(y.shape) == 1)

    for i in range(rows):
        for j in range(cols):

            if(np.isnan(X[:,i,j]).any()):
                M[i,j] = np.nan
            else:
                x_ij = np.copy(X[:,i,j])
                
                if(y_is_series):
                    y_ij = np.copy(y)
                else:
                    y_ij = np.copy(y[:,i,j])
                
                b = np.polyfit(x_ij,y_ij,1)

                M[i,j] = np.copy(b[0])

                del(b, x_ij, y_ij)

    return M
#===============================================================================================
#Getting a month out of a set of gridded data in an np.ndarray format
def selectMonth(x,month='dec'):
    monthDict = {'jan':0,'feb':1,'mar':2,'apr':3,'may':4,'jun':5,'jul':6, 'aug':7, 'sep':8,'oct':9,'nov':10,'dec':11}
    monthDict.update({'january':0,'february':1,'march':2,'april':3,'may':4,'june':5,'july':6, 'august':7, 'september':8,'october':9,'november':10,'december':11})

    choice = monthDict[month]
    nyrs = int(len(x)/12)
    y = np.copy(x)
    if(len(x.shape)==1):
        y.shape = (nyrs,12)
        y = y[:,choice]
    elif(len(x.shape)==2):
        space = y.shape[1]
        y.shape = (nyrs,12,space)
        y = y[:,choice,:]
        del(space)
    elif(len(x.shape)==3):
        rows = y.shape[1]
        cols = y.shape[2]
        y.shape = (nyrs,12,rows,cols)

        y = y[:,choice,:,:]

        del(rows,cols)
    return y

#===============================================================================================
#selecting the seasons from array gridded data
def selectSeason(x,season='winter'):
    seasonKeys = ['winter','spring','summer','fall','djf','mam','jja','son']
    winter = ['dec','jan','feb']
    spring = ['mar','apr','may']
    summer = ['jun','jul','aug']
    fall = ['sep','oct','nov']
    seasonList = [winter,spring,summer,fall]
    seasonDict = {seasonKeys[i]:seasonList[i%4] for i in range(8)}

    choice = seasonDict[season]

    y = np.zeros(selectMonth(x,'dec').shape)

    for i in range(3):
        y = y + selectMonth(x,choice[i])

    y = y/3
    del(choice)
    return y

#===============================================================================================
#Function that removes ONLY the climatology from the grid (Less versatile the removeGridCycle)
def removeGridClimatology(SST,clim = []):
    A = np.copy(SST)
    yrs = int(A.shape[0]/12)

    if(len(clim) == 0):
        C = np.copy(SST)
        C.shape = (yrs,12,C.shape[1],C.shape[2])
        C = np.mean(C,axis=0)

    else:
        C = np.copy(clim)    

    A.shape = (yrs,12,A.shape[1],A.shape[2])

    for i in range(12):
        A[:,i,:,:] = A[:,i,:,:] - C[i,:,:]

    A.shape = (12*A.shape[0],A.shape[2],A.shape[3])

    return A

#===============================================================================================
#Function that is designed to handle nan's in a gridded data set

def handleNAN(X):
    (time,rows,cols) = X.shape
    Y = np.copy(X)
    for i in range(rows):
        for j in range(cols):
            y_ij = np.copy(Y[:,i,j])
            if(np.isnan(y_ij).any()):
                Y[:,i,j] = np.nan

    return Y


#===============================================================================================
#Function that removes the seasonal Cycle at every single grid point in a data set
#Takes into account the shape of the matrix
def removeGridCycle(X):
    S = np.copy(X)
    npoly = 0
    #Trying a new method: the one I copied from Tim and everything hahahahaha
    (time,rows,cols) = S.shape
    
    nyrs = int(time/12)

    climo = np.copy(S[:(nyrs*12),:,:])
    climo.shape = (nyrs,12,rows,cols)
    climo = np.mean(climo,axis=0)

    ind = np.arange(time) % 12

    for m in range(12):
        k = ind == m
        S[k,:,:] = S[k,:,:] - climo[m,:,:]
    return S
#==========================================================================================================
#Function that calculated the climatology for a gridded data set
def gridClimatology(S):

    CLIM = np.copy(S)

    (time,rows,cols) = CLIM.shape

    years = int(time/12)

    CLIM.shape = (years,12,rows,cols) 


    CLIM = np.mean(CLIM,axis=0)   

    return CLIM
#==========================================================================================================
#A function the removes the trend from every grid point of a given data set
def detrendRegion(X,npoly=1,byMonth=True):
    S = np.copy(X)
    
    (time,rows,cols) = S.shape
    yrs = int(time/12)
    if(byMonth):
        S.shape = (yrs,12,rows,cols)
    ind = np.arange(time)
    for i in range(rows):
        for j in range(cols):
            if(byMonth):

                for m in range(12):
                    if(np.isnan(S[:,m,i,j]).any() ):
                        S[:,m,i,j] = np.isnan
                    else:
                        S[:,m,i,j] = st.detrend(S[:,m,i,j],npoly)
            else:
                if(np.isnan(S[:,i,j]).any() ):
                    S[:,i,j] = np.nan

                else:
                    S[:,i,j] = st.detrend(S[:,i,j],npoly)



    S.shape = (time,rows,cols)

    return S

#==========================================================================================================
#Function that uses the scipy.interpolate package adjusting the resoltuion of a gridded data set
#Note: This only works for data in a (lat,lon) format. If gridded data contains time as a dimension, use
#adjustAllRes below
def adjustRes(X,rowVals=[],colVals=[],newRows=32,newCols=40,method='nearest'):
    (rows,cols) = X.shape
    Y = np.copy(X)


    if(len(rowVals)==0 or len(colVals)==0):
        rowV = np.linspace(0,1,rows)
        colV = np.linspace(0,1,cols)

        gridRowV = np.linspace(0,1,newRows)
        gridColV = np.linspace(0,1,newCols)

    else:
        rowV = np.copy(rowVals)
        colV = np.copy(colVals)

        gridRowV = np.linspace(min(rowVals),max(rowVals),newRows)
        gridColV = np.linspace(min(colVals),max(colVals),newCols)


    rowOnes = np.ones(rows)
    colOnes = np.ones(cols)

    rowV = np.outer(rowV,colOnes)
    colV = np.outer(rowOnes,colV)

    rowV.shape = (rows*cols,)
    colV.shape = (rows*cols,)

    gridRowOnes = np.ones(newRows)
    gridColOnes = np.ones(newCols)

    gridRowV = np.outer(gridRowV,gridColOnes)
    gridColV = np.outer(gridRowOnes,gridColV)

    gridRowV.shape = (newRows*newCols,)
    gridColV.shape = (newRows*newCols,)

    del(rowOnes,colOnes,gridColOnes,gridRowOnes)

    Y.shape = (rows*cols)

    gridY = spi.griddata((rowV,colV),Y,(gridRowV,gridColV),method=method)


    gridY.shape = (newRows,newCols)
    gridRowV = np.reshape(gridRowV, (newRows,newCols))[:,0]
    gridColV = np.reshape(gridColV, (newRows,newCols))[0,:]

    if (len(rowVals)==0 or len(colVals)==0):
        return gridY    
    else: 
        return gridY, gridRowV, gridColV        
#====================================================================================================
#Function that changes the grid. This is slightly more general than the adjustRes
# Thhis also can be used to chhange data from an irregular grid onto a regular grid
def changeGrid(Z,x,y,new_x,new_y,method='linear'):

    rows = len(x)
    cols = len(y)
    x_grid = np.outer(x,np.ones(cols))
    y_grid = np.outer(np.ones(rows), y)


    new_rows = len(new_x)
    new_cols = len(new_y)
    new_x_grid = np.outer(new_x,np.ones(new_cols))
    new_y_grid = np.outer(np.ones(new_rows), new_y)

    Z_old = np.copy(Z)

    x_grid.shape = (rows*cols,)
    y_grid.shape = (rows*cols,)
    new_x_grid.shape = (new_rows*new_cols,)
    new_y_grid.shape = (new_rows*new_cols,)

    if(len(Z.shape) == 2):
        Z_old.shape = (rows*cols,)
    
    Z_new = spi.griddata((x_grid, y_grid), Z_old, (new_x_grid,new_y_grid), method=method)

    Z_new.shape = (new_rows, new_cols)

    return Z_new

#====================================================================================================
#Function for changing the grid of a series in (time, lat,lon) format
def changeGridSeries(Z,x,y,new_x,new_y,method='linear'):
    time = Z.shape[0]
    Z_new = np.zeros((time,len(new_x),len(new_y)))

    for t in range(time):
        Z_new[t,:,:] = changeGrid(Z[t,:,:],x,y,new_x,new_y,method)

    return Z_new
#====================================================================================================
#A function to adjust gridded data in a (time,lat,lon) format
def adjustAllRes(SST,rowVals,colVals,newRows=32,newCols=40,method='nearest'):
    S = np.copy(SST)
    time = S.shape[0]
    Snew = np.zeros((time,newRows,newCols))

    for t in range(time):
        Snew[t,:,:], newRow, newCol = adjustRes(S[t,:,:],rowVals,colVals,newRows,newCols,method)

    return Snew, newRow, newCol

#====================================================================================================
#Function that has the root mean squared for a region relative to climatology
def regionRMS(Stest,Scheck):
    print('Not really sure if I want to do this studpid thing')

    RMS = (Stest - Scheck) * (Stest - Scheck)

    RMS_clim = Stest * Stest

    return RMS / RMS_clim
#====================================================================================================
#function for regressing a general trend, like the global mean, from gridded data
def regressTrend(X,trend,deg=1):
    (nt,nx,ny) = X.shape
    X_det = np.zeros((nt,nx,ny))
    for i in range(nx):
        for j in range(ny):
            if(np.isnan(X[0,i,j] )):
                X_det[:,i,j] = np.nan
            else:
                x_ij = np.copy(X[:,i,j])
                b = np.polyfit(trend,x_ij,deg)
                nb = len(b)
                x_hat = np.zeros(nt)
                for k in range(len(b)):
                    x_hat = x_hat + b[k] * trend ** (nb-1-k)
                X_det[:,i,j] = x_ij - x_hat

    return X_det
