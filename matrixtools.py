import numpy as np
import numpy.linalg as nlg
import statstools as st
import scipy.linalg as slg
import numpy.random as nr

'''
Collection of functions for use in statisticall analysis of gridded data sets.
    
'''


#Getting the Empirical Orthhogonal Function (EOF) decomposition of a dataset in a (time, lat,lon) format
#Input:
#   X (np.ndarray): The data to decompose for EOFs in a (time, space) format,
#   k (int): The number of EOFs selected from the dataset (default 1)
#   weight (np.ndarray): A matrix of weights for each grid of the data set
#   includeVarExp (bool): variable that determines whhether or not to include the variance explained by each of the 
#       EOFs (default: True)
#Output:
#   F (np.ndarray): set of all of the principal component time series' for the entire data set. In this case,
#       each column of F represents a PC time series 
#   E (np.ndarray): set of the Empirical Orthogonal
def eof(X,k=1,weight=[],includeVarExp=True):
    try:

        S = np.copy(X)
        if(len(X.shape)==3):
            (time,rows,cols) = X.shape
            S.shape = (time,rows*cols)

        if(weight==[]):
            W = np.ones(S.shape[1])
        else:
            W = np.copy(weight) ** .5
            W.shape = (W.shape[0]*W.shape[1],)
        #Scaling everything by the weight matrix now
        S = S*W
        nonNum = np.isnan(S)
        ocn = np.isnan(S[0]) == False
        #Now,checking this beast for any non number values
        if(nonNum.any()):
            S = S[:,np.isnan(S[0,:])==False]

        
        #SVD of A
        [U,d,V] = slg.svd(S)

        #Getting the degrees of freedom
        a = np.sqrt(S.shape[0]-1)

        D = np.zeros(S.shape)

        for i in range(len(d)):
            D[i,i] = d[i]

        F = U*a

        E = np.dot(D,V)/a

        F = F[:,:k]

        E = E[:k,:]/W[ocn]
        
        if(includeVarExp):
            varexp = d[:k]*d[:k]/np.dot(d,d)
            return F,E,varexp
        else:
            return F,E
    
    except nlg.LinAlgError:
        print('Error: SVD did not converge')
#==========================================================================================================
#Function that construct the basic components of the Linear Inverse Model
#Input:
#   X: the matrx of time series' being used to construct the LIM
#   lag: the length of the training lag used in construction the LIM
#Output:
#   Dicionary with thhe following items
#       C0: the zeros-lag covariance matrix of the data.
#       C_tau: the lagged covariance matrix of the given training lag.
#       Green: The Green function of the LIM
#       L: The operator matrix of the LIM.
#       Eigenmodes: The eigenvalues of the operator matrix L
#       POPs: The Eigenvectors of the operator matrix, where the i'th column represents a specific matrix
#   
def lim(X,lag):

    #Getting the variables of the data and everything
    C = np.copy(X)
    time = C.shape[0]
    valid = np.isnan(C) == False
    C = C[valid]
    space = int(len(C)/time)
    C.shape = (time,space)
    
   
    x0 = np.copy(X[:(time-lag),:])
    xl = np.copy(X[lag:,:])


    (time,dim) = x0.shape


    C_0 = np.dot(x0.T,x0)/(time-lag) + 1 - 1 
    C_lag = np.dot(xl.T,x0)/(time-lag) + 1 - 1

    
    if(nlg.det(C_0)==0 or nlg.cond(C_0)>1e8):
        print('Not a good choice of things')
        G = np.dot(C_lag,nlg.pinv(C_0))

    else:
        C_01 = nlg.inv(C_0) + 1 - 1
        G = np.dot(C_lag,C_01)

    #The elusive linear operator
    L = slg.logm(G)/lag
    
    #The green function here
    Gr = lambda t: G ** (t/lag)

    Q = -(np.dot(L,C_0.T) + np.dot(C_0,L.T))

    #Getting the eigenmodes and principal oscillation patterns
    [modes,pops] = nlg.eig(L)

    return {'c0':C_0,'c_tau':C_lag,'green':Gr,'l':L,'q':Q,'eigenmodes':modes, 'pops':pops}

#================================================================================
#Function defining the Mutual Information Criteria to be associated with the
# multivariate linear regression. The purpose of this function is to determine the optimal 
# number of EOFs 
def mic_mlr(X,Y):
    space_x = np.sum(np.logical_not(np.isnan(X[0,:,:])))
    space_y = np.sum(np.logical_not(np.isnan(Y[0,:,:])))
    time  = X.shape[0]

    xmax = np.min([space_x,time])
    ymax = np.min([space_y,time])

    [Fx,Ex,vx] = eof(X,xmax)
#Function for Multiple Linear regression of the data X and Y (EOFs deterined by
# mic_gaussian)
def mlr(X,Y):
    space_x = np.sum(np.logical_not(np.isnan(X[0,:,:])))
    space_y = np.sum(np.logical_not(np.isnan(Y[0,:,:])))
    time  = X.shape[0]

    xmax = np.min([space_x,time])
    ymax = np.min([space_y,time])

    ntot = X.shape[0] 
    XtX = np.dot(X.T,X)/ntot + 1 - 1 
    XtX = nlg.inv(XtX) + 1 - 1

    XtY = np.dot(X.T,Y)/ntot + 1 - 1

    B = np.dot(XtX,XtY) + 1 - 1

    return B

#==================================================================================
#FUNCTIONS THE conducting Canonical Correlation Analysis (CCA) OF A GIVEN DATA SET

#Mutual information criteria associated with CCA. This is run inside of the cca function
#   below.
#Input:
#   Fx: The pricipal components of the X data in a (time, eof) format
#   Fy: The pricipal components of the Y data in a (time,eof) format
#   xmax: The maximum number of EOFs to check for X (defautl is 15)
#   ymax: The maximum number of EOFs to check for Y (default is 15)
#Output:
#   M: The matrix of the mutual information criteria in an (nx,ny) format
def mic_cca(Fx,Fy,xmax,ymax):
    n = Fx.shape[0]
    dof = n - 1

    M = np.zeros((xmax,ymax))
    #Going through and checking for overfitting
    for i in range(1,xmax+1):
        Fx_i = Fx[:,:i]
        for j in range(1,ymax+1):
            Fy_j = Fy[:,:j]

            Sxy = np.dot(Fx_i.T, Fy_j)/dof
            rho = nlg.svd(Sxy)
            rho = rho[1]

            try:

                #The penalty term nows
                p = (n+1)*((i+j)/(n-i-j-2) - i/(n-i-2) - j/(n-j-2) )

                mic = np.sum(np.log(1-rho*rho)) + p

                M[(i-1),(j-1)] = mic
            except ZeroDivisionError:
                print('Error: One of the denominators in the penalty term is equal to 0')
                break
    

    return M
 

#Function that conducts Canonical Correlation Analysis on two datasets

def cca(X,Y, lon_x,lat_x,lon_y,lat_y):
    
    #Input:
    #   X: The X dataset
    #   Y: the y dataset
    #   lon_x: Longituse of the X data set
    #   lat_x: Latitude of the X data set
    #   lon_y: Longitude of the Y data set
    #   lat_y: Latitude of the Y data set

    #Output (dictionary with the following keys):
    #   Rx: The variates of the X coordinate
    #   Ry: The variates of the Y data set
    #   Px: The loading vectors of the X data set
    #   Py: The loading vectors of the Y data set
    #   rho: The correlations associated with each variate and loading vector pairing
    #   nx: The number of EOFs used for X
    #   ny: the number of EOFs used for Y

    try:
        makeArea = lambda lon,lat: np.outer(np.cos(lat*np.pi/180),np.ones(len(lon)))

        space_x = np.sum(np.logical_not(np.isnan(X[0,:,:])))
        space_y = np.sum(np.logical_not(np.isnan(Y[0,:,:])))
        time  = X.shape[0]

        xmax = np.min([space_x,time,30])
        ymax = np.min([space_y,time,30])

        area_x = makeArea(lon_x,lat_x)
        area_y = makeArea(lon_y,lat_y)

        [Fx,Ex] = eof(X-np.mean(X,axis=0),xmax,area_x,False)

        [Fy,Ey] = eof(Y-np.mean(Y,axis=0),ymax,area_y,False)

        M = mic_cca(Fx,Fy,xmax,ymax)

        nx,ny = np.where(M==np.min(M))
        nx = nx[0] + 1
        ny = ny[0] + 1

        dof = X.shape[0] - 1
        

        #Truncating the EOFs based on the mutual information criteria
        Fx = Fx[:,:nx]
        Ex = Ex[:nx,:]

        Fy = Fy[:,:ny]
        Ey = Ey[:ny,:]

        #Getting the covariance matrix and its SVD to calculate the variates and the loading vectors
        S_xy = np.dot(Fx.T, Fy)/dof

        
        [Qx,rho,Qy] = nlg.svd(S_xy)

        #Getting thhe variates associated with CCA
        Rx = np.dot(Fx,Qx)
        Ry = np.dot(Fy,Qy)

        #Now, gathering the matrix of loading vectors
        Px = np.zeros((nx,X.shape[1],X.shape[2])) * np.nan
        Py = np.zeros((ny,Y.shape[1],Y.shape[2])) * np.nan



        makeOcn = lambda V: np.logical_not(np.isnan(V[0,:,:]))

        ocn_x = makeOcn(X)
        ocn_y = makeOcn(Y)

        Px[:,ocn_x] = np.dot(Qx.T,Ex)
        Py[:,ocn_y] = np.dot(Qy.T,Ey)

        #Px = np.dot(Qx.T,Ex)
        #Py = np.dot(Qy.T,Ey)

        return {'Rx':Rx, 'Ry':Ry, 'Px':Px,'Py':Py, 'rho':rho , 'nx':nx, 'ny':ny, 'mic':M}
        
    except nlg.LinAlgError:
        print('Error: The SVD of the covariance matrix did not converge.')