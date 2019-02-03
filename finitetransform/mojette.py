# -*- coding: utf-8 -*-
"""
Python module for computing methods related to the Mojette transform.

The transform (resulting in projections) is computed via the 'transform' member.

Assumes coordinate system with rows as x-axis and cols as y-axis. Thus angles are taken as complex(q,p) with q in the column direction.
Use the farey module to generate the angle sets.

Created on Tue Aug 26 09:59:54 2014

@author: uqscha22
"""
import finitetransform.farey as farey #local module
import finitetransform.radon as radon
import numpy as np
import scipy.fftpack as fftpack
import pyfftw

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

def projectionLength(angle, P, Q):
    '''
    Return the number of bins for projection at angle of a PxQ image.
    Wraps function from Farey module
    '''
    return farey.projectionLength(angle, P, Q) #no. of bins
    
def toFinite(fareyVector, N):
    '''
    Return the finite vector corresponding to the Farey vector provided for a given modulus/length N
    and the multiplicative inverse of the relevant Farey angle
    Wraps function from Farey module
    '''
    return farey.toFinite(fareyVector, N)
    
def finiteTranslateOffset(fareyVector, N, P, Q):
    '''
    Translate offset required when mapping Farey vectors to finite angles
    Returns translate offset and perp Boolean flag pair
    Wraps function from Farey module
    '''
    return farey.finiteTranslateOffset(fareyVector, N, P, Q)

def isKatzCriterion(P, Q, angles, K = 1):
    '''
    Return true if angle set meets Katz criterion for exact reconstruction of
    discrete arrays
    '''
    sumOfP = 0
    sumOfQ = 0
    n = len(angles)
    for j in range(0, n):
        p, q = farey.get_pq(angles[j])
        sumOfP += abs(p)
        sumOfQ += abs(q)
        
#    if max(sumOfP, sumOfQ) > max(rows, cols):
    if sumOfP > K*P or sumOfQ > K*Q:
        return True
    else:
        return False

def project(image, q, p, dtype=np.int32):
    '''
    Projects an array at rational angle given by p, q.
    
    Returns a list of bins that make up the resulting projection
    '''
    offsetMojette = 0
    rows, cols = image.shape
    totalBins = abs(q)*(rows-1) + abs(p)*(cols-1) + 1
#    print "Projection (%d, %d) has %d bins" % (q, p, totalBins)
    
    if q*p >= 0: #If positive slope
        offsetMojette = p*(rows-1)

    projection = np.zeros(totalBins, dtype)
    for x in range(0, rows):
        for y in range(0, cols):
            if q*p >= 0:
                translateMojette = q*y - p*x + offsetMojette #GetY = q, GetX = p
            else:
                translateMojette = p*x - q*y; #GetY = q, GetX = p
#            print "t:", translateMojette, "x:", x, "y:", y, "p:", p, "q:", q
            projection[translateMojette] += image[x,y]
    
    return projection
    
def transform(image, angles, dtype=np.int32, prevProjections = []):
    '''
    Compute the Mojette transform for a given angle set and return list of projections.
    
    The angle set is assumed to be a list of 2-tuples as (q, p). Returns a list of projections.
    Previous projections can be provided (for use in iterative reconstruction methods), but must be of correct size.
    '''
    mu = len(angles)

    #Compute Mojette
    projections = []
    for n in range(0, mu):
        p = int(angles[n].imag)
        q = int(angles[n].real)
        projection = project(image, q, p, dtype)
        if not prevProjections:
            projections.append(projection)
        else:
            projections.append(projection+prevProjections[n])
        
    return projections

def backproject(projections, angles, P, Q, norm = True, dtype=np.int32, prevImage = np.array([])):
    '''
    Directly backprojects (smears) a set of projections at rational angles given by angles in image space (PxQ).
    
    Returns an image of size PxQ that makes up the reconstruction
    '''
    image = np.zeros((P,Q),dtype)

    normValue = 1.0
    if norm:
        normValue = float(len(angles))
    
    for projection, angle in zip(projections, angles):
        p = int(angle.imag)
        q = int(angle.real)
        offsetMojette = 0
        if q*p >= 0: #If positive slope
            offsetMojette = p*(Q-1)
        for x in range(0, P):
            for y in range(0, Q):
                if q*p >= 0:
                    translateMojette = q*y - p*x + offsetMojette #GetY = q, GetX = p
                else:
                    translateMojette = p*x - q*y #GetY = q, GetX = p
    #            print "t:", translateMojette, "x:", x, "y:", y, "p:", p, "q:", q
                prevValue = 0
                if prevImage.size > 0:
                    prevValue = prevImage[x,y]
                    
                try:
                    image[x,y] += projection[translateMojette]/normValue + prevValue
                except IndexError:
                    image[x,y] += 0 + prevValue
    
    return image

def backprojectPadded(projections, angles, P, Q, N, M, norm = True, centered=True, dtype=np.int32):
    '''
    Directly backprojects (smears) a set of projections at rational angles given by angles in image space (PxQ).
    
    Returns an image of size PxQ that makes up the reconstruction
    '''
    image = np.zeros((N,M),dtype)

    normValue = 1.0
    if norm:
        normValue = float(len(angles))
    
    centerX = 0
    centerY = 0    
    if centered:
        centerX = N/2 - Q/2
        centerY = M/2 - P/2
    
    for projection, angle in zip(projections, angles):
        p = int(angle.imag)
        q = int(angle.real)
        offsetMojette = 0
        if q*p >= 0: #If positive slope
            offsetMojette = p*(M-1)
            
        #pad projection accordingly
        projectionPadded = np.zeros(projectionLength(angle,N,M))
        #copy projection
        for index, binValue in enumerate(projection):
            if q*p >= 0:
                x = (Q-1)+centerX
                y = centerY
                translateMojette = int(q*y - p*x + offsetMojette) #GetY = q, GetX = p
            else:
                x = centerX
                y = centerY
                translateMojette = int(p*x - q*y) #GetY = q, GetX = p
#            print "t:", translateMojette, "x:", x, "y:", y, "p:", p, "q:", q, "offset", offsetMojette, "index:", index
            projectionPadded[translateMojette+index] = binValue
        
        for x in range(0, N):
            for y in range(0, M):
                if q*p >= 0:
                    translateMojette = q*y - p*x + offsetMojette #GetY = q, GetX = p
                else:
                    translateMojette = p*x - q*y #GetY = q, GetX = p
#                print "t:", translateMojette, "x:", x, "y:", y, "p:", p, "q:", q, "offset", offsetMojette
                
                try:
                    image[x,y] += projectionPadded[translateMojette]/normValue
                except IndexError:
                    image[x,y] += 0
    
    return image
    
def backprojectFast(projections, angles, P, Q, N, norm = True, dtype=np.int32):
    '''
    Fast backprojects (smears) a set of projections at rational angles given by angles in Fourier space (NxN).
    
    Returns an image of size PxQ that makes up the reconstruction
    '''
    fftImage = np.zeros((N,N), dtype=np.complex)    
    
    for projection, angle in zip(projections, angles):
        frtProjection = finiteProjection(projection, angle, P, Q, N)
        fftProjection = fftpack.fft(frtProjection, N)
        m, inv = farey.toFinite(angle, N)
        radon.setSlice(m, fftImage, fftProjection)
    
    image = fftpack.ifft2(fftImage)
    
    return np.real(image)
    
def finiteProjection(projection, angle, P, Q, N, center=False):
    '''
    Convert a Mojette projection taken at angle into a finite (FRT) projection.
    '''
    dyadic = True
    if N % 2 == 1: # if odd, assume prime
        dyadic = False    
    shiftQ = int(N/2.0+0.5)-int(Q/2.0+0.5)
    shiftP = int(N/2.0+0.5)-int(P/2.0+0.5)
    
    finiteProj = np.zeros(N)
    p, q = farey.get_pq(angle)
    m, inv = farey.toFinite(angle, N)
#    print "p:", p, "q:", q, "m:", m, "inv:", inv
    translateOffset, perp = farey.finiteTranslateOffset(angle, N, P, Q)
    angleSign = p*q    
    
    if dyadic:    
        for translate, bin in enumerate(projection):
            if angleSign >= 0 and perp: #Reverse for perp
                translateMojette = translateOffset - translate
            else:
                translateMojette = translate - translateOffset
            translateFinite = (inv*translateMojette)%N
            if center:
                translateFinite = (translateFinite + shiftQ + m*(N-shiftP))%N
            finiteProj[translateFinite] += bin
    else:
        for translate, bin in enumerate(projection):
            if angleSign >= 0 and perp: #Reverse for perp
                translateMojette = int(translateOffset) - int(translate)
            else:
                translateMojette = int(translate) - int(translateOffset)
                
            if translateMojette < 0:
                translateFinite = ( N - ( inv*abs(translateMojette) )%N )%N
            else:         
                translateFinite = (inv*translateMojette)%N #has issues in C, may need checking
            if center:
                translateFinite = (translateFinite + shiftQ + m*(N-shiftP))%N
            finiteProj[translateFinite] += bin
        
    return finiteProj

#inversion methods
def toDRT(projections, angles, N, P, Q, center=False):
    '''
    Convert the Mojette (asymetric) projection data to DRT (symetric) projections.
    Use the iFRT to reconstruct the image. Requires N+1 or N+N/2 projections if N is prime or dyadic respectively.
    Returns the resulting DRT space as a 2D array
    '''
    size = int(N + N/2)
    dyadic = True
    if N % 2 == 1: # if odd, assume prime
        size = int(N+1)
        dyadic = False
        
    m = 0
    
    frtSpace = np.zeros( (size,N) )
    
    if dyadic:
        print("Dyadic size not tested yet.")
        #for each project
        '''for index, proj in enumerate(projections):
            p, q = farey.get_pq(angles[index])
            
            m, inv = farey.toFinite(angles[index], N)

            frtSpace[m][:] = finiteProjection(proj, angles[index], P, Q, N, center)'''
        
    else: #prime size
        for index, proj in enumerate(projections):
            p, q = farey.get_pq(angles[index])
            
            m, inv = farey.toFinite(angles[index], N)

            frtSpace[m][:] = finiteProjection(proj, angles[index], P, Q, N, center)
    
    return frtSpace

def nConjugateGradient(n, imageEstimate, projMeasured, projAngles, convergenceData=False):
    '''
    Conjugate Gradient Method
    perform "n" iterations of CGM
    
    see http://en.wikipedia.org/wiki/Conjugate_gradient_method
    conjugate gradient method to solve B.P.x = B.m
    where B = backprojection, P = projection, x = imageObjective, and m = projMeasured
    use x = imageEstimate [which may be zero] as initial guess in B.P.x = B.m
    
    If convergenceData is true then a list is also returned with convergence info
    '''
    P, Q = imageEstimate.shape
    imageIN = np.zeros(imageEstimate.shape,np.float32)
    # reconstruct projections to get B.m
    imageResidual = backproject(projMeasured,projAngles,P,Q,False,np.float)
    # calulate residual, r = x - B.P.x
    projEstimate = transform(imageEstimate,projAngles)
    imageOUT = backproject(projEstimate,projAngles,P,Q,False,np.float)
    imageResidual -= imageOUT
    # assign residual to imageIN
    imageIN += imageResidual
    normCurrResidual = np.sum(imageResidual*imageResidual)
#    normInitResidual = normCurrResidual
    residualsList = []
    residualsList.append(normCurrResidual)
    for i in np.arange(n):
        # calculate imageOUT = B.P.imageIN
#        projEstimate = []
        projEstimate = transform(imageIN,projAngles,np.float)
#        imageOUT = 0
        imageOUT = backproject(projEstimate,projAngles,P,Q,False,np.float)
        # update x and r
        alpha = normCurrResidual/np.sum(imageOUT*imageIN)
        imageEstimate += alpha*imageIN
        imageResidual -= alpha*imageOUT
        # update imageIN
        normPrevResidual = normCurrResidual
        normCurrResidual =  np.sum(imageResidual*imageResidual)
        residualsList.append(normCurrResidual)
#        if(i==0):
#          normInitResidual = normCurrResidual
        beta = normCurrResidual/normPrevResidual
        imageIN *= beta
        imageIN += imageResidual
        
    if convergenceData:
        return imageEstimate, residualsList
    else:
        return imageEstimate

def nFastConjugateGradient(n, N, imageEstimate, projMeasured, projAngles, convergenceData=False):
    '''
    Conjugate Gradient Method
    perform "n" iterations of CGM
    
    see http://en.wikipedia.org/wiki/Conjugate_gradient_method
    conjugate gradient method to solve B.P.x = B.m
    where B = backprojection, P = projection, x = imageObjective, and m = projMeasured
    use x = imageEstimate [which may be zero] as initial guess in B.P.x = B.m
    
    If convergenceData is true then a list is also returned with convergence info
    '''
    P, Q = imageEstimate.shape
    imageIN = np.zeros(imageEstimate.shape,np.float32)
    # reconstruct projections to get B.m
    imageResidual = backprojectFast(projMeasured,projAngles,P,Q,N,False,np.float)
    # calulate residual, r = x - B.P.x
    projEstimate = transform(imageEstimate,projAngles)
    imageOUT = backprojectFast(projEstimate,projAngles,P,Q,N,False,np.float)
    imageResidual -= imageOUT
    # assign residual to imageIN
    imageIN += imageResidual
    normCurrResidual = np.sum(imageResidual*imageResidual)
#    normInitResidual = normCurrResidual
    residualsList = []
    residualsList.append(normCurrResidual)
    for i in np.arange(n):
        # calculate imageOUT = B.P.imageIN
#        projEstimate = []
        projEstimate = transform(imageIN,projAngles,np.float)
#        imageOUT = 0
        imageOUT = backprojectFast(projEstimate,projAngles,P,Q,N,False,np.float)
        # update x and r
        alpha = normCurrResidual/np.sum(imageOUT*imageIN)
        imageEstimate += alpha*imageIN
        imageResidual -= alpha*imageOUT
        # update imageIN
        normPrevResidual = normCurrResidual
        normCurrResidual =  np.sum(imageResidual*imageResidual)
        residualsList.append(normCurrResidual)
#        if(i==0):
#          normInitResidual = normCurrResidual
        beta = normCurrResidual/normPrevResidual
        imageIN *= beta
        imageIN += imageResidual
        
    if convergenceData:
        return imageEstimate, residualsList
    else:
        return imageEstimate
        
def mlem(iterations, P, Q, g_j, angles, projector, backprojector, epsilon=1e-6, dtype=np.int32):
    '''
    # Gary's implementation
    # From Lalush and Wernick;
    # f^\hat <- (f^\hat / |\sum h|) * \sum h * (g_j / g)          ... (*)
    # where g = \sum (h f^\hat)                                   ... (**)
    #
    # self.f is the current estimate f^\hat
    # The following g from (**) is equivalent to g = \sum (h f^\hat)
    '''
    norm = True
    fdtype = np.float32
#    weighting = backprojector(np.ones_like(g_j, dtype), angles, P, Q, norm, dtype)
    weighting = np.ones((P,Q), fdtype)
#    print "weighting shape:", weighting.shape
#    print "weighting:", weighting
    f = np.ones((P,Q), fdtype)
    
    for i in range(0, iterations):
        g = projector(f, angles, dtype)
#        print "g:", g
    
#        for x in range(len(angles)):
#            g[x].clip(min=epsilon, out=g[x])
#        print "g':", g
    
        # form parenthesised term (g_j / g) from (*)
        r = []
        for x in range(len(angles)):
            r.append(g_j[x].astype(float) / g[x].astype(float)) #float for avoiding truncation
#        print "r:", r
    
        # backproject to form \sum h * (g_j / g)
        g_r = backprojector(r, angles, P, Q, norm, fdtype)
#        print "gr shape:", g_r.shape
#        print "gr", g_r
    
        # Renormalise backprojected term / \sum h)
        # Normalise the individual pixels in the reconstruction
        f *= g_r / weighting
        
    return f
        
def deconvolvePSF(backProj, angles, P, Q, M, pad=0):
    '''
    Use a PSF based deconvolution of the backprojected Mojette projections.
    See Svalbe 2014 on Backprojection Filtration.
    '''
    normPSF = True
    if pad == 0:
        pad = 2*M
    
    #create PSF 
    psfImage = psf(angles, P, Q, normPSF, pad) #4 times larger PSF
    #print "PSF:", psfImage
    print("PSF Size:", psfImage.shape)
    
    #create weights and regularise
    print("Regularise PSF")
    Tpn, D, psfWeightsImage = psfWeights(psfImage, angles, P, Q, pad, normPSF)
    psfRegImage = np.multiply(psfImage, psfWeightsImage)
    #psfRegImage = np.multiply(psfImage, Tpn)
    
    #deconvolve PSF
    print("Deconvolve")
    backProjDeconv = radon.deconvolve(backProj, psfRegImage)
    
    return backProjDeconv
    
def deconvolveFinitePSF(backProj, angles, P, Q, M, pad=0):
    '''
    Use a PSF based deconvolution of the backprojected Mojette projections.
    See Svalbe 2014 on Backprojection Filtration.
    '''
    normPSF = True
    if pad == 0:
        pad = 2*M
    
    #create PSF 
    psfImage = psfFinite(angles, P, Q, normPSF, pad) #4 times larger PSF
    #print "PSF:", psfImage
    print("PSF Size:", psfImage.shape)
    
    #create weights and regularise
    print("Regularise PSF")
    Tpn, D, psfWeightsImage = psfWeights(psfImage, angles, P, Q, pad, normPSF)
    psfRegImage = np.multiply(psfImage, psfWeightsImage)
    #psfRegImage = np.multiply(psfImage, Tpn)
    
    #deconvolve PSF
    print("Deconvolve")
    backProjDeconv = radon.deconvolve(backProj, psfRegImage)
    
    return backProjDeconv

#helper functions
def discreteSliceSamples(angle, b, fftShape):
    '''
    Generate the b points along slice at angle of DFT space with shape.
    '''
    r, s = fftShape
    q = farey.getX(angle)
    p = farey.getY(angle)
    u = []
    v = []
    
    u.append(0 + r/2)
    v.append(0 + s/2)
    for m in range(1, b/4):
         u.append(p*m + r/2)
         v.append(q*m + s/2)
    for m in range(-b/4, 1):
         u.append(p*m + r/2)
         v.append(q*m + s/2)
#    print "u:",u
#    print "v:",v
    return u, v
    
def sliceSamples(angle, b, fftShape, center=False):
    '''
    Generate the b points along slice at angle of DFT space with shape.
    '''
    r, s = fftShape
    p, q = farey.get_pq(angle)
    u = []
    v = []
    offsetU = 0
    offsetV = 0
    if center:
        offsetU = r/2
        offsetV = s/2
#    increment = 1.0/math.sqrt(p**2+q**2)
    
    u.append(0 + offsetU)
    v.append(0 + offsetV)
    for m in range(1, (b-1)/2):
         u.append((1.0/p)*m + offsetU)
         v.append(-(1.0/q)*m + offsetV)
    for m in range(-(b-1)/2, 1):
#        print "m:", m, "delP:", -(1.0/p)*m + offsetU, "delQ:", (1.0/q)*m + offsetV
        u.append((1.0/p)*m + offsetU)
        v.append(-(1.0/q)*m + offsetV)
#    print "u:",u
#    print "v:",v
    return u, v

#PSFs
import math 
    
def psf(angles, P, Q, norm = True, pad=0, center=True):
    '''
    Create point spread function for mojette angles given as given by work of 
    Svalbe 2014: Back-Projection Filtration Inversion of Discrete Projections
    '''
    mu = len(angles)
    Nx = Q
    Ny = P
    if pad > 0:
        Nx = pad
        Ny = pad
    centerX = 0
    centerY = 0
    if center:
        centerX = Nx/2
        centerY = Ny/2
    normValue = 0
    if norm:
        normValue = -1.0/(mu-1.0)

    projections = []
    for angle in angles:
        B = farey.projectionLength(angle, Ny, Nx)
        p, q = farey.get_pq(angle)
        offsetMojette = 0
        #need to compute where center pixel has to be for back projection consistency
        if q*p >= 0: #If positive slope
            offsetMojette = p*(Nx-1)
        if q*p >= 0:
            translateMojette = int(q*centerY - p*centerX + offsetMojette) #GetY = q, GetX = p
        else:
            translateMojette = int(p*centerX - q*centerY) #GetY = q, GetX = p
        projection = np.zeros(B, dtype=np.float32)
        projection[:] = normValue
        projection[translateMojette] = 1.0
#        print projection
        projections.append(projection)
    '''image = np.zeros((Q,P)) #inefficient but easier to understand method
    image[Q/2, P/2] = 1.0
    projections = transform(image, angles, dtype=np.float)'''
        
    return backproject(projections, angles, Ny, Nx, False, dtype=np.float)/mu
    
def psfFinite(angles, P, Q, norm = True, pad=0, center=True):
    '''
    Create point spread function for mojette angles given as given by work of 
    Svalbe 2014: Back-Projection Filtration Inversion of Discrete Projections
    This version creates a periodic PSF based on size pad. Pad should be prime for the FRT.
    '''
    mu = len(angles)
    Nx = Q
    Ny = P
    if pad > 0:
        Nx = pad
        Ny = pad
    centerX = 0
    centerY = 0
    if center:
        centerX = Nx/2
        centerY = Ny/2
    normValue = 0
    if norm:
        normValue = -1.0/(mu-1.0)

    projections = []
    for angle in angles:
        B = farey.projectionLength(angle, Ny, Nx)
        p, q = farey.get_pq(angle)
        offsetMojette = 0
        #need to compute where center pixel has to be for back projection consistency
        if q*p >= 0: #If positive slope
            offsetMojette = p*(Nx-1)
        if q*p >= 0:
            translateMojette = q*centerY - p*centerX + offsetMojette #GetY = q, GetX = p
        else:
            translateMojette = p*centerX - q*centerY #GetY = q, GetX = p
        projection = np.zeros(B, dtype=np.float32)
        projection[:] = normValue
        projection[translateMojette] = 1.0
#        print projection
        projections.append(projection)
    
    drtSpace = toDRT(projections, angles, pad, pad, pad, False)
        
    return radon.ifrt(drtSpace, pad, False, False)/mu

import scipy.signal as signal
    
def psfWeights(psfImage, angles, P, Q, pad=0, norm = True):
    '''
    Mask used to encapsulate the correctly backprojected region and image support region that can be utilised in regularising the PSF.
    Based on PxQ image and the angle set (needs to be symmetric), create a bounded circular region weights for regularising the PSF.
    '''
    Nx = Q
    Ny = P
    if pad > 0:
        Nx = pad
        Ny = pad
    centerX = Nx/2
    centerY = Ny/2
    
    #compute image of positive PSF values
    if norm:
        pImage = psfImage >= -1e-6 #>= 0, but epsilon
    else:
        pImage = psfImage > 0 
    nImage = np.invert(pImage)
    Tpn = signal.correlate2d(pImage.astype(float), nImage.astype(float), mode='same')
    maxValue = Tpn.max()
    Tpn /= maxValue
    
    #support
    radius = math.sqrt((Q+1)**2+(P+1)**2)/2.0 #1 pixel bigger
    region = np.zeros((Nx,Ny), dtype=np.bool)
    for i, row in enumerate(psfImage):
        for j, col in enumerate(row):
            distance = math.sqrt( (i-float(centerX))**2 + (j-float(centerY))**2)
#            if ((i-float(centerX)) <= Q/2 and (i-float(centerX)) >= -Q/2) and ((j-float(centerY)) <= P/2 and (j-float(centerY)) >= -P/2):
            if distance <= radius:
#                print i, j
                region[i, j] = True
                
    D = signal.correlate2d(region.astype(float), region.astype(float), mode='same')
#    D = signal.convolve(region.astype(float), region.astype(float), mode='same')
    maxValue = D.max()
    D /= maxValue
    #Gaussian
    '''D = np.zeros((Nx,Ny), dtype=np.float)
    for i, row in enumerate(psfImage):
        for j, col in enumerate(row):
            D[i, j] = np.exp( -((i-float(centerX))**2 + (j-float(centerY))**2)/(4.0*radius**2) )'''
    
    weights = np.multiply(Tpn, D)
    maxValue = weights.max()
    weights /= maxValue
    
    #reset correctly backprojected region
    #compute max radius for correctly backprojected region
    maxAngle = np.abs( np.array(angles) ).max() #max angle length
    print("max angle:", maxAngle)
    correctRadius = int(abs(maxAngle)+0.5) #round up
    print("Corrected Radius:", correctRadius)
    for i, row in enumerate(weights):
        for j, col in enumerate(row):
            distance = math.sqrt( (i-float(centerX))**2 + (j-float(centerY))**2)
            if distance <= correctRadius:
#                print i, j
                weights[i, j] = 1.0
                Tpn[i, j] = 1.0
                
    return Tpn, D, weights
    
def psfRegularise(psfImage, angles, P, Q, pad=0):
    '''
    Regularise the PSF (inplace) to allow deconvolution based mojette inversion.
    See Svalbe 2014.
    ''' 
#    epsilon = 1e-6
    Tpn, D, weights = psfWeights(psfImage, angles, P, Q, pad)
    
    #threshold
    '''psfImageThreshold = np.zeros(psfImage.shape)
    psfImageThresholdDash = np.zeros(psfImage.shape)
    for i, row in enumerate(psfImage):
        for j, col in enumerate(row):
            if psfMaskImage[i, j] > 0 and psfImage[i, j] >= epsilon:
                psfImageThreshold[i, j] = 1
            if psfMaskImageDash[i, j] > 0 and psfImage[i, j] >= -epsilon:
#                print i, j, psfImage[i, j]
                psfImageThresholdDash[i, j] = 1'''
                
    return np.multiply(psfImage, weights)

#angle sets
def angleSet_ProjectionLengths(angles, P, Q):
    '''
    Returns a matching list of projection lengths for each angle in set
    '''
    binLengthList = []
    for angle in angles:
        binLengthList.append(projectionLength(angle,P,Q))
        
    return binLengthList

def angleSet_Finite(p, quadrants=1, finiteList=False):
    '''
    Generate the minimal L1 angle set for the MT that has finite coverage.
    If quadrants is more than 1, two quadrants will be used.
    '''
    fareyVectors = farey.Farey()
    
    octants = 2
    if quadrants > 1:
        octants = 4
    if quadrants > 2:
        octants = 8
        
    fareyVectors.compactOn()
    fareyVectors.generate(p/2, octants)
    vectors = fareyVectors.vectors
    sortedVectors = sorted(vectors, key=lambda x: x.real**2+x.imag**2) #sort by L2 magnitude
    
    #compute corresponding m values
    finiteAngles = []
    for vector in sortedVectors:
        if vector.real == 0:
            m = 0
        elif vector.imag == 0:
            m = p
        else:
            m, inv = toFinite(vector, p)
        finiteAngles.append(m)
#        print("m:", m, "vector:", vector)
#    print("sortedVectors:", sortedVectors)
    #print(finiteAngles)
        
    #ensure coverage
    count = 0
    filled = [0]*(p+1) #list of zeros        
    finalVectors = []
    finalFiniteAngles = [] 
    for vector, m in zip(sortedVectors, finiteAngles):
        if filled[m] == 0:
            count += 1
            filled[m] = 1
            finalVectors.append(vector)
            finalFiniteAngles.append(m)
            
        if count == p+1:
            break
    
    if finiteList:
        return finalVectors, finalFiniteAngles
        
    return finalVectors

def angleSet_MinimalL1(P, Q, octant=0, binLengths=False, K = 1):
    '''
    Generate the minimal L1 angle set for the MT.
    Parameter K controls the redundancy, K = 1 is minimal.
    If octant is non-zero, full quadrant will be used. Octant schemes are as follows:
        If octant = -1, the opposing octant is also used.
        If octant = 0, (default), only use one octant.
        If octant = 1, use one quadrant. Octant will be mirrored from diagonal to form a quadrant.
        If octant = 2, quadrant will be mirrored from axis to form a half.
    Function can also return bin lengths for each bin.
    '''
    angles = []
    fareyVectors = farey.Farey()
    maxPQ = max(P,Q)
        
    fareyVectors.compactOn()
    fareyVectors.generate(maxPQ-1, 1)
    vectors = fareyVectors.vectors
    sortedVectors = sorted(vectors, key=lambda x: x.real+x.imag) #sort by L1 magnitude
    
    index = 0
    binLengthList = []
    angles.append(sortedVectors[index])
    binLengthList.append(projectionLength(sortedVectors[index],P,Q))
    while not isKatzCriterion(P, Q, angles, K) and index < len(sortedVectors): # check Katz
        index += 1
        angles.append(sortedVectors[index])
        p, q = farey.get_pq(sortedVectors[index]) # p = imag, q = real
        
        binLengthList.append(projectionLength(sortedVectors[index],P,Q))
        
#        if isKatzCriterion(P, Q, angles):
#            break
        
        if octant == 0:
            continue
        
        #add octants
        if octant == -1:
            nextOctantAngle = farey.farey(p, -q) #mirror from axis
            angles.append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
        if octant > 0 and p != q:
            nextOctantAngle = farey.farey(q, p) #swap to mirror from diagonal
            angles.append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
        if octant > 1:
            nextOctantAngle = farey.farey(p, -q) #mirror from axis
            angles.append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
            if p != q: #dont replicate
                nextOctantAngle = farey.farey(q, -p) #mirror from axis and swap to mirror from diagonal
                angles.append(nextOctantAngle)
                binLengthList.append(projectionLength(nextOctantAngle,P,Q))
    
    if octant > 1: #add the diagonal and column projections when symmetric (all quadrant are wanted)
        nextOctantAngle = farey.farey(1, 0) #mirror from axis
        angles.append(nextOctantAngle)
        binLengthList.append(projectionLength(nextOctantAngle,P,Q))
    
    if binLengths:
        return angles, binLengthList
    return angles
    
def angleSet_Symmetric(P, Q, octant=0, binLengths=False, K = 1):
    '''
    Generate the minimal L1 angle set for the MT.
    Parameter K controls the redundancy, K = 1 is minimal.
    If octant is non-zero, full quadrant will be used. Octant schemes are as follows:
        If octant = -1, the opposing octant is also used.
        If octant = 0,1 (default), only use one octant.
        If octant = 2, octant will be mirrored from diagonal to form a quadrant.
        If octant = 4, 2 quadrants.
        If octant = 8, all quadrants.
    Function can also return bin lengths for each bin.
    '''
    angles = []
    fareyVectors = farey.Farey()
    maxPQ = max(P,Q)

    fareyVectors.compactOff()
    fareyVectors.generate(maxPQ-1, 1)
    vectors = fareyVectors.vectors
    sortedVectors = sorted(vectors, key=lambda x: x.real**2+x.imag**2) #sort by L2 magnitude
    
    index = 0
    binLengthList = []
    angles.append(sortedVectors[index])
    binLengthList.append(projectionLength(sortedVectors[index],P,Q))
    while not isKatzCriterion(P, Q, angles, K) and index < len(sortedVectors): # check Katz
        index += 1
        angles.append(sortedVectors[index])
        p, q = farey.get_pq(sortedVectors[index]) # p = imag, q = real
        
        binLengthList.append(projectionLength(sortedVectors[index],P,Q))
        
#        if isKatzCriterion(P, Q, angles):
#            break
        
        if octant == 0:
            continue
        
        #add octants
        if octant == -1:
            nextOctantAngle = farey.farey(p, -q) #mirror from axis
            angles.append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
        if octant > 0 and p != q:
            nextOctantAngle = farey.farey(q, p) #swap to mirror from diagonal
            angles.append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
        if octant > 1:
            nextOctantAngle = farey.farey(p, -q) #mirror from axis
            angles.append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
            if p != q: #dont replicate
                nextOctantAngle = farey.farey(q, -p) #mirror from axis and swap to mirror from diagonal
                angles.append(nextOctantAngle)
                binLengthList.append(projectionLength(nextOctantAngle,P,Q))
    
    if octant > 1: #add the diagonal and column projections when symmetric (all quadrant are wanted)
        nextOctantAngle = farey.farey(1, 0) #mirror from axis
        angles.append(nextOctantAngle)
        binLengthList.append(projectionLength(nextOctantAngle,P,Q))
    
    if binLengths:
        return angles, binLengthList
    return angles
    
def angleSet_Entropic(P, Q, octant=0, binLengths=False, K = 1):
    '''
    Generate the minimal L1 angle set for the MT.
    Parameter K controls the redundancy, K = 1 is minimal.
    If octant is non-zero, full quadrant will be used. Octant schemes are as follows:
        If octant = -1, the opposing octant is also used.
        If octant = 0,1 (default), only use one octant.
        If octant = 2, octant will be mirrored from diagonal to form a quadrant.
        If octant = 4, 2 quadrants.
        If octant = 8, all quadrants.
    Function can also return bin lengths for each bin.
    '''
    angles = []
    fareyVectors = farey.Farey()
    maxPQ = max(P,Q)

    fareyVectors.compactOff()
    fareyVectors.generate(maxPQ-1, 1)
    vectors = fareyVectors.vectors
    sortedVectors = sorted(vectors, key=lambda x: x.real**2+x.imag**2, reverse=True) #sort by L2 magnitude
    #reverse set
    
    index = 0
    binLengthList = []
    angles.append(sortedVectors[index])
    binLengthList.append(projectionLength(sortedVectors[index],P,Q))
    while not isKatzCriterion(P, Q, angles, K) and index < len(sortedVectors): # check Katz
        index += 1
        angles.append(sortedVectors[index])
        p, q = farey.get_pq(sortedVectors[index]) # p = imag, q = real
        
        binLengthList.append(projectionLength(sortedVectors[index],P,Q))
        
#        if isKatzCriterion(P, Q, angles):
#            break
        
        if octant == 0:
            continue
        
        #add octants
        if octant == -1:
            nextOctantAngle = farey.farey(p, -q) #mirror from axis
            angles.append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
        if octant > 0 and p != q:
            nextOctantAngle = farey.farey(q, p) #swap to mirror from diagonal
            angles.append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
        if octant > 1:
            nextOctantAngle = farey.farey(p, -q) #mirror from axis
            angles.append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
            if p != q: #dont replicate
                nextOctantAngle = farey.farey(q, -p) #mirror from axis and swap to mirror from diagonal
                angles.append(nextOctantAngle)
                binLengthList.append(projectionLength(nextOctantAngle,P,Q))
    
    if octant > 1: #add the diagonal and column projections when symmetric (all quadrant are wanted)
        nextOctantAngle = farey.farey(1, 0) #mirror from axis
        angles.append(nextOctantAngle)
        binLengthList.append(projectionLength(nextOctantAngle,P,Q))
    
    if binLengths:
        return angles, binLengthList
    return angles
    
def angleSubSets_Symmetric(s, mode, P, Q, octant=0, binLengths=False, K = 1):
    '''
    Generate the minimal L1 angle set for the MT for s subsets.
    Parameter K controls the redundancy, K = 1 is minimal.
    If octant is non-zero, full quadrant will be used. Octant schemes are as follows:
        If octant = -1, the opposing octant is also used.
        If octant = 0,1 (default), only use one octant.
        If octant = 2, octant will be mirrored from diagonal to form a quadrant.
        If octant = 4, 2 quadrants.
        If octant = 8, all quadrants.
    Function can also return bin lengths for each bin.
    '''
    angles = []
    subsetAngles = []
    for i in range(s):
        subsetAngles.append([])
    fareyVectors = farey.Farey()
    maxPQ = max(P,Q)

    fareyVectors.compactOff()
    fareyVectors.generate(maxPQ-1, 1)
    vectors = fareyVectors.vectors
    sortedVectors = sorted(vectors, key=lambda x: x.real**2+x.imag**2) #sort by L2 magnitude
    
    index = 0
    subsetIndex = 0
    binLengthList = []
    angles.append(sortedVectors[index])
    subsetAngles[subsetIndex].append(sortedVectors[index])
    binLengthList.append(projectionLength(sortedVectors[index],P,Q))
    while not isKatzCriterion(P, Q, angles, K) and index < len(sortedVectors): # check Katz
        index += 1
        angles.append(sortedVectors[index])
        subsetAngles[subsetIndex].append(sortedVectors[index])
        p, q = farey.get_pq(sortedVectors[index]) # p = imag, q = real
        
        binLengthList.append(projectionLength(sortedVectors[index],P,Q))
        
#        if isKatzCriterion(P, Q, angles):
#            break
        
        if octant == 0:
            continue
        
        #add octants
        if octant == -1:
            nextOctantAngle = farey.farey(p, -q) #mirror from axis
            angles.append(nextOctantAngle)
            subsetAngles[subsetIndex].append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
            if mode == 1:
                subsetIndex += 1
                subsetIndex %= s
        if octant > 0 and p != q:
            nextOctantAngle = farey.farey(q, p) #swap to mirror from diagonal
            angles.append(nextOctantAngle)
            subsetAngles[subsetIndex].append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
            if mode == 1:
                subsetIndex += 1
                subsetIndex %= s
        if octant > 1:
            nextOctantAngle = farey.farey(p, -q) #mirror from axis
            angles.append(nextOctantAngle)
            subsetAngles[subsetIndex].append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
            if mode == 1:
                subsetIndex += 1
                subsetIndex %= s
            if p != q: #dont replicate
                nextOctantAngle = farey.farey(q, -p) #mirror from axis and swap to mirror from diagonal
                angles.append(nextOctantAngle)
                subsetAngles[subsetIndex].append(nextOctantAngle)
                binLengthList.append(projectionLength(nextOctantAngle,P,Q))
                if mode == 1:
                    subsetIndex += 1
                    subsetIndex %= s
                
        if mode == 0:
            subsetIndex += 1
            subsetIndex %= s
    
    if octant > 1: #add the diagonal and column projections when symmetric (all quadrant are wanted)
        nextOctantAngle = farey.farey(1, 0) #mirror from axis
        angles.append(nextOctantAngle)
        subsetAngles[0].append(nextOctantAngle)
        binLengthList.append(projectionLength(nextOctantAngle,P,Q))
    
    if binLengths:
        return angles, subsetAngles, binLengthList
    return angles, subsetAngles
    
def angleSetSliceCoordinates(angles, P, Q, N, center=False):
    '''
    Compute the 2D coordinates of each translate (in NxN DFT space) of every projection having angle in angles.
    Returns a list of u, v coordinate arrays [[u_0[...],v_0[...]], [u_1[...],v_1[...]], ...] per angle
    '''
    coords = []
    translateOffset = 0
    translateMojette = 0
    translateFinite = 0
    m = 0

    offset = 0.0
    if center:
        offset = N/2.0
    
    for index, angle in enumerate(angles):
        u = []
        v = []
        coordinateList = []
        p = int(angle.imag)
        q = int(angle.real)
        angleSign = p*q
        
        m, inv = farey.toFinite(angle, N)
        translateOffset, perp = farey.finiteTranslateOffset(angle, N)
        B = projectionLength(angle, P, Q)
        
        for translate in range(0, B):
            if angleSign >= 0 and perp: #Reverse for perp
                translateMojette = translateOffset - translate
            else:
                translateMojette = translate - translateOffset
            
            translateFinite = (inv*translateMojette)%N #has issues in C, may need checking
#            frtSpace[m][translateFinite] += bin
            u.append( (translateFinite+offset)%N )
            v.append( (m*translateFinite+offset)%N )
        
        coordinateList.append(u)
        coordinateList.append(v)
        coords.append(coordinateList)
    
    return coords
