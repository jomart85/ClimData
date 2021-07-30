import limtools as lt
import matrixtools as mt
import numpy as np
import numpy.linalg as nlg
import regiontools as rt



#============================================================================================================
#Getting the RMS for the predicted time series
def seriesRMStest(X,eofs,lon,lat,globe=[],removeGlobe=True,npoly=2):
    byMonth = False
    S = np.copy(X)
    sst = np.copy(globe)
    A = np.outer(np.cos(lat*np.pi/180), np.ones(len(lon)))

    #S = rt.annualScaleRegion(S)
    #sst = ts.annualScale(sst)
    if(removeGlobe and (len(sst)==S.shape[0]) ):
        S = rt.removeGlobe(S,sst)
    else:
        if( (npoly<0) and (len(sst)==S.shape[0])):
            S = rt.regressTrend(S,sst,-npoly)
        else:
            S = rt.detrendRegion(S,byMonth,npoly)

    S_make = np.copy(S)
    #S = rt.annualScaleRegion(rt.removeGridCycle(S))
    (time,rows,cols) = S.shape
    ocn = np.isnan(S[0,:,:]) == False

    [F,E,v] = mt.eof(S,eofs,A,True)
    (time,rows,cols) = S.shape
    #(time,space) = F.shape
    test = 10

    #climTot = np.mean(ts.climatology(amvForClim))


    #Building a method that will end up doing shit
    step = 1
    start = 1
    stop = start + test


    rms = np.zeros(test+1)
    rmsClim = np.zeros(test+1)
    #The Root mean square via persistence
    #amv = ts.annualScale(ts.removeCycle(er.erTimeSeries(iyst,iynd,latBounds=latBds)))
    t = 0
    allClim = np.array([])
    #A set of the least damped modes
    ldModes = np.array([])

    while(stop <= time):
        
        #OLD WAY
        #Making the testing data for a while
        #Stest = np.append(S_make[:(start*12),:,:],S_make[(stop*12):,:,:])
        #Stest.shape = ((time-test)*12,rows,cols)
        #The climatology for the test data thing and whatever
        #testClim = rt.gridClimatology(Stest)
        #Stest = rt.annualScaleRegion(rt.removeGridClimatology(Stest,testClim))
        #Perturbing the whole thing

        #NEW WAY
        Stest = np.append(S_make[:(start),:,:],S_make[(stop):,:,:])
        Stest.shape = ((time-test),rows,cols)
        #The climatology for the test data thing and whatever
        testClim = np.mean(Stest,axis=0)
        Stest = Stest - testClim
        #Perturbing the whole thing

        [Ftest,Etest] = mt.eof(Stest,eofs,A,False)

        #New way of getting the new climatology
        Scheck = S_make[(start):(stop),:,:]
        Scheck = Scheck - testClim        
        
        limR = mt.lim(Ftest,1)

        L = limR['l']
        Q = limR['q']
        what = (nlg.eigvals(Q)>0).all() # True #
        if(what):
            Eproj = Etest#EcheckE#
            F0 = Ftest[start-1,:]
            pred = lt.predictLIM(L,F0,[A],[Eproj],[eofs],[S[0,:,:]],test+1,1)
            
            p = pred['timeseries'][0] #ts.annualScale(pred['timeseries'][0])

            #sCheck = amv[start:stop]
            sCheck = np.dot(Scheck[:,ocn],A[ocn])/np.sum(A[ocn])
            #Getting the initial value from the data
            s0 = np.dot(Stest[start-1,ocn],A[ocn])/np.sum(A[ocn])
            sCheck = np.append(s0,sCheck)
            
            rms = rms + (sCheck - p) * (sCheck - p)

            #Putting in the climatology for the test thing I guess
            
            rmsClim = rmsClim + sCheck * sCheck

            #allClim = np.append(allClim,climTest)
            t += 1

        start += step
        stop += step

    #RMS = RMS[1:]
    #RMS_clim = RMS_clim[1:]

    r = 3
    c = 3


    rms = np.sqrt(rms/t)
    rmsClim = np.sqrt(rmsClim/t)

    rms = rms/rmsClim

    return rms
#============================================================================================================
#Getting the rms for the gridded data
def rmsRegionTest(X,eofs,lon,lat,globe=[],pullGlobe=True,npoly=2,smoothWindow=0):
    byMonth = False
    S = np.copy(X)
    sst = np.copy(globe)
    A = np.outer(np.cos(lat*np.pi/180), np.ones(len(lon)))
    #THIS IS THE NEW FORMAT WE'RE TESTING OUT
    #S = rt.annualScaleRegion(S)
    #sst = ts.annualScale(sst)
    if(pullGlobe and (len(sst)==S.shape[0]) ):
        S = rt.removeGlobe(S,sst)
    else:
        if( (npoly<0) and (len(sst)==S.shape[0])):
            S = rt.regressTrend(S,sst,-npoly)
        else:
            S = rt.detrendRegion(S,byMonth,npoly)

    S_make = np.copy(S)
    #THIS WAS THE ORIGINAL FORMAT


    ocn = np.isnan(S[0,:,:]) == False
    #S_anoms = rt.annualScaleRegion(rt.removeGridCycle(S))
    [F,E,v] = mt.eof(S,eofs,A,True)

    #(time,space) = F.shape
    test = 10
    (time,rows,cols) = S.shape
    #Building a method that will end up doing shit
    step = 1
    start = 1
    stop = test + start #test + season

    RMS = np.zeros((3,S.shape[1],S.shape[2]))
    RMS_clim = np.zeros((3,S.shape[1],S.shape[2]))
    rms = np.zeros((test+1,S.shape[1],S.shape[2]))
    rmsClim = np.zeros((test+1,S.shape[1],S.shape[2]))

    #A TEMPORARY THING FOR CLIMATOLOGY
    CLIM = np.array([])
    #amv = ts.annualScale(ts.removeCycle(er.erTimeSeries(iyst,iynd,latBounds=latBds)))

    t = 0
    starts = [1,3,6] #[0,0,0]#
    stops = [3,6,9] #[1,1,1]#


    while(stop <= time):
        #OLD WAY; REMOVING CLIMATOLOGY
        #Stest = np.append(S_make[:(start*12),:,:],S_make[(stop*12):,:,:])
        #Stest.shape = ((time-test)*12,rows,cols)
        #testClim = rt.gridClimatology(Stest)
        #Stest = rt.annualScaleRegion(rt.removeGridClimatology(Stest,testClim))

        #NEW WAY: REMOVING THE MEAN STATE FROM THE DATA
        Stest = np.append(S_make[:(start),:,:],S_make[(stop):,:,:])
        Stest.shape = ((time-test),rows,cols)
        
        #If smoothwindow is positive, then place a running average onto the smoothed out data and everything
        S0 = Stest - np.mean(Stest,axis=0)
        [Frough,Erough] = mt.eof(S0,eofs,A,False)
        F0 = np.copy(Frough[start-1,:])
        del(Frough,Erough,S0)
        #Now, if it's smooth, you take the thing out
        if(smoothWindow>0):
            Stest = rt.runningAverageRegion(Stest,smoothWindow)
            
            
            
        
        testClim = np.mean(Stest,axis=0)
        Stest = Stest - testClim

        #Stest = np.append(S[:start,:,:],S[stop:,:,:])
        #Stest.shape = ((time-test),rows,cols)
        [Ftest,Etest] = mt.eof(Stest,eofs,A,False)


        #NEW WAY: REMOVING THE MEAN STATE
        Scheck = S_make[(start):(stop),:,:]
        Scheck = Scheck - testClim


        limR = mt.lim(Ftest,1)

        L = limR['l']
        Q = limR['q']
        what =True # (nlg.eigvals(Q)>0).all()
        if(what):
            #init = int(start/12)
            Eproj = Etest
            #F0 = Ftest[start-1,:]#Fcheck[0,:] F[start-1,:] #
            pred = lt.predictLIM(L,F0,[A],[Eproj],[eofs],[S[0,:,:]],test+1,1/step)
            P = np.copy(S[:(test+1),:,:])
            P[:,ocn] = np.copy(pred['space'][0]).real# rt.annualScaleRegion(pred['space'][0])

            Scheck = np.append(S[(start-1),:,:],Scheck)
            Scheck.shape = (test+1,rows,cols)

            #Getting the series rms thing or whatever
            rms = rms + (P - Scheck) * (P - Scheck)
            rmsClim = rmsClim + Scheck * Scheck

            for j in range(RMS.shape[0]):
                a = starts[j]
                b = stops[j]
                Pj = np.mean(P[a:b,:,:],axis=0)
                Sj = np.mean(Scheck[a:b,:,:],axis=0)

                RMS[j,:,:] = RMS[j,:,:] + (Pj  - Sj) * (Pj - Sj)

                RMS_clim[j,:,:] = RMS_clim[j,:,:] + Sj*Sj


            t += 1

        start += step
        stop += step
    #RMS = RMS[1:]
    #RMS_clim = RMS_clim[1:]


    RMS = np.sqrt(RMS/t)

    RMS_clim = np.sqrt(RMS_clim/t)

    RMS = RMS/RMS_clim

    rms = np.sqrt(rms/t)
    rmsClim = np.sqrt(rmsClim/t)

    rms = rms/rmsClim
    rms = np.dot(rms[:,ocn],A[ocn])/np.sum(A[ocn])


    return RMS
#============================================================================================================
#Getting larger diagnostics of the LIM for further examination of properties
def limDiagnostics(X,eofs,lon,lat,globe=[],removeGlobe=False,npoly=2,smoothWindow=0):
    byMonth = False
    S = np.copy(X)
    sst = np.copy(globe)
    A = np.outer(np.cos(lat*np.pi/180), np.ones(len(lon)))

    #S = rt.annualScaleRegion(S)
    #sst = ts.annualScale(sst)
    if(removeGlobe and (len(sst)==S.shape[0]) ):
        S = rt.removeGlobe(S,sst)
    else:
        if( (npoly<0) and (len(sst)==S.shape[0])):
            S = rt.regressTrend(S,sst,-npoly)
        elif(npoly == 0):
            S = S - np.mean(S,axis=0)
        else:
            S = rt.detrendRegion(S,byMonth,npoly)
    

    #If smoothWindow is positive, place a running average on your dataset after detrending
    if(smoothWindow>0):
        S = rt.runningAverageRegion(S,smoothWindow)

    S = S - np.nanmean(S,axis=0)
    
    trainLag = 1
    
    #This was the original, just checking the newer format to make sure this is okay no matter what
    #==============================================================
    #S = rt.removeGridCycle(S)
    #Putting the SST to an annual scale
    #S = rt.annualScaleRegion(S)
    #==============================================================
    Aocn = A[np.isnan(S[0,:,:])==False]

    #yrs = ht.getYears(iyst,iynd)

    N = 1

    [F,E,varExp] = mt.eof(S,eofs,A,includeVarExp=True)

    EOF = np.copy(S[:eofs,:,:])
    ocn = np.logical_not(np.isnan(S[0,:,:]))
    for i in range(eofs):
        EOF[i,ocn] = np.copy(E[i])

    limResults = mt.lim(F,trainLag)

    L = limResults['l']
    Q = limResults['q']
    modes = limResults['eigenmodes']

    things = lt.eigenmodes(L,F,[E],[eofs],[A],[S[0,:,:],['Atlantic']])


    ocn = np.isnan(S[0,:,:]) == False
    amv = np.dot(S[:,ocn],A[ocn])/np.sum(A[ocn])

    return {'modeProj':things, 'sst':S, 'amv':amv, 'modes':modes,'noise':Q, 'variance':varExp,'eof':EOF}