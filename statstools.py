'''
Set of functions usefule for statistical analysis on a time series
'''
import numpy as np
import numpy.fft as fft
import numpy.linalg as nlg
import scipy.stats as sps
import scipy.optimize as spo
import scipy.fftpack as spf
import timeseriestools as ts



#====================================================================================
#Function that gets a regression function onto a dataset

def regression( y, npoly=1):
   
    time = np.arange(0,len(y))
    #For now, only doing the linear case. More will be added
    yhat = np.zeros(len(y))
    #This function is only necessary for fitting an exponential function
    expFunc = lambda x,a,b,c : a * np.exp(b*x) + c
    if(npoly==2.72 or (npoly == np.exp(1)) or (npoly == 'e')):
        b = np.polyfit([time[0],time[-1]],np.log([y[0], y[-1] ]),1)
        popt, pcov = spo.curve_fit(expFunc, time, y, p0 = (np.exp(b[1]), b[0], 1 ))
    else:
        b = np.polyfit(time,y,deg=npoly)
        
    

    

    if((npoly==2.72) or (npoly == np.exp(1)) or (npoly == 'e') ):
        yhat = expFunc(time,*popt)

    else:
        max_deg = len(b) - 1
        yhat = b[max_deg]

        for i in range(max_deg):
            yhat = yhat + (b[max_deg -1 - i] * (time**(i+1)))
   
    return yhat


#====================================================================================
#Function that removes a trend from a dataset
def detrend(vector, npoly=1):

    y = regression(vector,npoly)

    return vector - y

#====================================================================================
#Function used to change a data set so that it has a mean of 0 and a standard deviation
#of 1
def standardize(vector):
    stdev = np.std(vector)

    if(stdev == 0):
        return np.ones(len(vector))
    else:
        return (vector-np.mean(vector))/stdev


#====================================================================================
#Function that calculated the power spectrum of a given data set
def powerSpectrum(vector,k,normal=True):

    if(normal):
        v = standardize(vector)
    else:
        v = np.copy(vector)
    pgram = fft.fft(v)

    ntot = len(pgram)

    freq = np.arange(ntot)/ntot

    pgram = (pgram*pgram.conj())/len(v)

    pgram = pgram.real

    stop = int(round(ntot/2))

    pgram = pgram[:stop]

    freq = freq[:stop]

    pgram = ts.runningAverage(pgram,k)
    

    freq = freq[k:(len(freq)-k)]
    #Changing to the period
    #freq = 1/freq
    #freq = np.arange(1,(len(pgram)+1))/ntot
    return [freq,pgram]

#====================================================================================
#A power spectrum for a red noise process, for comparison
def pspec_redNoise(phi,n,k):
    k = 0
    x0 = 1
    x = np.array([x0])
    for i in range(n-1):
        xi = phi*x[i] + k
        x = np.append(x,xi)

    [frn,psrn] = powerSpectrum(x,k)
    return [frn,psrn]

#====================================================================================
#Function that gets the confidence interval for points along a power spectrum
def pspecCI(pspec,k,alpha=.95):
    #Starting out with generating the chi-squared distribution
    dof = 2*(2*k+1)
    X2 = sps.chi2(dof)
    [b1,b2] = X2.interval(alpha)
    #Setting the bounds for the confidence interval
    numerator = 2*(2*k+1)
    lower = dof*pspec/b2
    upper = dof*pspec/b1
    return [lower,upper]

#====================================================================================
#Basic function for generating an AR-1 red noise process
def ar1(phi,n,x0=0,mu=0,sd=1, spinup=100,k=0):
    x = np.array([x0])
    k = 0
    for i in range(spinup + n-1):
        xi = phi*x[i] + np.random.normal(loc=mu,scale=sd) + k
        x = np.append(x,xi)

    return x[spinup:]

#====================================================================================
#Coefficient of an AR1 process fit to an input time series
def fitAR1(x,lag):

    n = len(x)

    k = n - lag

    x0 = x[:k]

    xl = x[lag:]

    phi = np.mean(x0*xl)/np.mean(x0*x0)

    phi = phi ** (1/lag)

    return phi

#====================================================================================
#An AR2 Process.
def ar2(phi1,phi2,n,x0=1,x1=1,k=0,mu=0,sd=1,spinup=100):
    x = np.array([x0,x1])

    for i in range(2,spinup+n):
        xt = phi1*x[i-1] - phi2*x[i-2] + np.random.normal(mu,sd) + k

        x = np.append(x,xt)

    return x[spinup:]

    
#========================================================================================
#Function that gets the lead-lag relationshsip between two variables
def leadLagCorr(x,y,lead=10,standard=True):
    
    #Seeing if x and y should be standardized
    if(standard):
        u = standardize(x)
        v = standardize(y)
    else:
        u = np.copy(x)
        v = np.copy(y)
    ntot = len(x)
    corrs = np.array([])
    #Start with x leading y
    for i in range(-lead,0):
        k = ntot + i
        c = np.corrcoef(u[:k],v[-i:])[0,1]
        corrs = np.append(corrs,c)

    #Now, have y lead x
    for j in range(0,lead+1):
        k = ntot - j
        c = np.corrcoef(u[j:],v[:k])[0,1]
        corrs = np.append(corrs,c)

    del(u,v,c,ntot,i,j)

    return corrs

#========================================================================================
#Function that gathers the r_squared of two variables
def get_Rsquared(x,y):
    r_squared = -500
    if((type(x) is tuple) or (type(x) is list)):
        X = np.ones((len(y),len(x)+1))
        for i, xi in enumerate(x):
            X[:,i] = np.copy(xi)
        
        XtX = np.dot(X.T,X)
        Xty = np.dot(X.T,y)

        if(nlg.cond(XtX)>1000):
            XtX_inv = nlg.pinv(XtX)
        else:
            XtX_inv = nlg.inv(XtX)
        b = np.dot(XtX_inv,Xty)

        yhat = np.squeeze(np.dot(X,b))
        mu_y = np.mean(y)

        r_squared = 1 - (np.sum( (y-yhat) * (y-yhat) )/np.sum( (y-mu_y) * (y-mu_y) ))
    elif(type(x) is np.ndarray):
        r = sps.pearsonr(x,y)[0]
        r_squared = r * r
    else:
        print('Invalid datatype for x')
    return r_squared