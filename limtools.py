import numpy as np
import numpy.linalg as nlg
import scipy as sp
import scipy.linalg as slg
import matrixtools as mt
import numpy.random as nr

'''
A set of functions that are build to assist with Linear Inverse Modeling
'''

#=======================================================================================================
#Function to build the X matrix based on the input lists of EOFs and PC time series given
#Assumes all of PC's are in a (time,neof) layout
def buildX(PC_list,neof_list):
    time = PC_list[0].shape[0]
    #Getting the EOF space from the input EOF list
    space = sum(neof_list)

    #Making the X matrix from the (time, eofspace) layout being used.

    X = np.zeros((time,space)) - 30000
    items = len(neof_list)

    start = 0
    stop = 0

    for i in range(items):
        stop += neof_list[i]

        X[:,start:stop] = np.copy(PC_list[i])

        start += neof_list[i]

    return X

#=======================================================================================================
#Function that takes the X matrix and decomposes it into a set of time series and maps (maps are for fun)
def projectX(X,EOF_list,neof_list,area_list,map_list,name_list=[]):
    #Input:
    #   X: the eof space collection that has to be mapped (time,eofspace)
    #   EOF_list: the set of EOF spaces for projecting
    #   area_list: the set of area weighting for time series construction
    #   map_lst: a list of the maps that the project EOF spaces can be placed onto and that
    #Output:
    #   ts_dictionary: a dictionary of the time series' constructed, named based on either an input of
    #       names or on numbers assigned
    #   map_dictionary: a dictionary or the time evolved maps, just for fun

    ntot = len(neof_list)
    time = X.shape[0]

    if(len(name_list)==0):
        keys = []
        for i in range(ntot):
            keys = keys + [str(i+1)]
    else:
        keys = name_list
    start = 0
    stop = 0
    #A temporary set of lists holding togethr all of the data and everything
    tsList = []
    mapList = []

    if(ntot > 1):
        for n in range(ntot):
            stop += neof_list[n]
            #Getting together the area, map, and eof
            A = area_list[n]

            R = map_list[n]
            vals = np.isnan(R) == False


            E = EOF_list[n]

            x = np.copy(X[:,start:stop])

            x = np.dot(x,E)
            #Building the maps out of this sort of shit and everything HA HA HA HA HA
            M = np.zeros((time,R.shape[0],R.shape[1]))
            for t in range(time):
                M[t,:,:] = np.copy(R)
                M[t,vals] = np.copy(x[t,:])

            x = np.dot(x,A[vals])/np.sum(A[vals])

            tsList = tsList + [x]
            mapList = mapList + [M]
                
            start += neof_list[n]

        map_dictionary = {keys[m]:mapList[m] for m in range(ntot)}

        ts_dictionary = {keys[s]: tsList[s] for s in range(ntot) }

        output = {'maps':map_dictionary, 'series':ts_dictionary}

    else:
            #Getting together the area, map, and eof
            n = 0
            A = area_list[n]

            R = map_list[n]
            vals = np.isnan(R) == False


            E = EOF_list[n]

            x = np.copy(X)

            x = np.dot(x,E)
            #Building the maps out of this sort of shit and everything
            M = np.zeros((time,R.shape[0],R.shape[1]))
            for t in range(time):
                M[t,:,:] = np.copy(R)
                M[t,vals] = np.copy(x[t,:])

            x = np.dot(x,A[vals])/np.sum(A[vals])

            output = {'maps':M,'series':x}
        

    return output

#=======================================================================================================
#Function that gathers the eigenmodes and their projections. Will not project to EOF space. That will be
#done separately
def eigenmodes(L,X,EOFlist,neofList,areaList,mapList,nameList=[]):

    #Getting the values and all of that shit and everything
    [modes,pops] = nlg.eig(L)

    time = X.shape[0]

    #Getting the adjoint of L
    #A = mt.adjoint(L)
    #[h,G] = nlg.eig(A)
    #del(h)
    G = nlg.inv(pops)
    i = 0
    ntot = len(modes)

    keyList = []

    modeList = []
    D = np.dot(G,X.T)
    while (i<ntot):
        k = str(i+1)
        v = pops[:,i]
        g = G[:,i]
        #Putting together the projection vector
        d = D[i,:]

        P = np.zeros((X.shape),dtype=complex)

        for t in range(time):
            P[t,:] = d[t]*v

        #Adding ththe conjugate if the current mode is complex
        '''
        if(modes[i].imag != 0 or (np.abs(modes[i]) >= 5e-14 )):

            #print('Complex Mode.')
            
            #print(i)

            k = k + '/' + str(i+2)

            v2 = pops[:,i+1]
            g2 = G[:,i+1]

            #Putting together the projection vector
            d2 = D[i+1,:]
            for t in range(time):
                P[t,:] = P[t,:] + d2[t]*v2

            i += 1
        '''

        keyList = keyList + [k]
        #pList = pList + [P]

        modeList = modeList + [projectX(P,EOFlist,neofList,areaList,mapList,nameList)]
        
        i += 1
        
    output = {keyList[z]:modeList[z] for z in range(len(keyList))}

    return output

#=======================================================================================================
#Projection time series for all of the eigenmodes and everything. FUCK!

#=======================================================================================================
#Funcion that actually runs the LIM for prediction
def predictLIM(L,y0,A_list,E_list,eofList,layout_list,time=1000,dt=1):
    '''
    The two separate methods that can be used in this evaluation
    lag: computes the lagged prediction based off of the data set being given
    iter: computes a 1-step lag from the previous point
    '''

    #Getting the Green Function from the given L
    Gr = lambda tau: slg.expm(L*tau)
    #Initializing the Y matrix that will be output
    #ndim will be used later
    ndim = L.shape[0]
    its = int(time/dt)

    Y = np.zeros((its,ndim))
    Y[0,:] = np.copy(y0)

    #Now, determing whatto be done wil the data at hand
    t = dt
    n = 1
    while(n<its):
        Y[n,:] = np.dot(Gr(t),y0)
        n += 1
        t +=dt

    #Now, running through the eof list and getting the data sets and everything
    #Looping through and everything
    start = 0
    stop = 0
    ntot = len(eofList)
    tsList = []
    spaceList = []
    if(ntot>1):
        for n in range(ntot):
            stop += eofList[n]
            #Getting the EOF projectstions and everything
            space = np.copy(Y[:,start:stop])
            space = np.dot(space,E_list[n])
            layout = layout_list[n]
            ocn = np.isnan(layout) == False
            layout[ocn] = np.copy(space)

            A = A_list[n]
            ts = np.dot(space,A[ocn])/np.sum(A[ocn])

            tsList = tsList + [ts]
            spaceList = spaceList + [layout]

            start += eofList[n]

    else:
        space = np.copy(Y)
        space = np.dot(space,E_list[0])
        layout = layout_list[0]
        ocn = np.isnan(layout) == False
        layout[ocn] = np.copy(space[0,:])

        A = A_list[0]
        ts = np.dot(space,A[ocn])/np.sum(A[ocn])

        tsList = [ts]
        spaceList = [space]
        #print('Under construction')

    #A dictionary of the outputs and everything
    output = {'space': spaceList, 'timeseries':tsList}
    return output

#=======================================================================================================
#Function that gets the least damped eigenmodes because doing it all by hand stinks
def leastDampedMode(L,X,EOFlist,neofList,areaList,mapList,nameList=[]):
    #Getting the collection of the eigenmodes and all of that crap
    things = eigenmodes(L,X,EOFlist,neofList,areaList,mapList,nameList)
    #Finding the modes and that crud
    modes = nlg.eigvals(L)

    #There are two options:
    #   1. it is an exponentiall decaying mode so there
    #   2: it is an osillatory mode

    modeIndex = np.arange(len(modes))

    #Getting the least damped mode and everything
    ldMode = np.max(modes.real)

    ldIndex = modeIndex[modes.real==ldMode]
    modeTS = np.zeros(X.shape[0],dtype=complex)
    returnMode = ''
    for i in range(len(ldIndex)):
        modeTerm = str(ldIndex[i] + 1)

        modeTS += things[modeTerm]['series']
        returnMode += modeTerm
        if (len(ldIndex)>1 and (i<(len(ldIndex)-1))):
            returnMode += '/'

    del(things)
    return [returnMode, modeTS.real]

    

#=======================================================================================================
#Function for integrating the LIM with the two-step method used in Penland Matrosova (1994)
def integrateLIM(L,Q,x0,time=1000,dt=1):
    X = np.copy(x0)

    #Getting the eigenvalue and the eigenvectors of Q
    (r,q) = nlg.eig(Q)
    r = np.sqrt(2*r*dt)
    q = q.T

    its = time/dt

    n = 1
    t = dt
    yt = np.copy(x0)
    ynext = np.zeros(len(x0))
    while(n<its):
        w = nr.normal(size=len(x0))
        ynext = yt + np.dot(L,yt)*2*dt + np.dot(r*w,q)
        x = (yt+ynext)/2
        X = np.append(X,x)
        t += dt
        n += 1
        yt = np.copy(ynext)

    X.shape = (n,len(x0))
    return X

#====================================================================================================