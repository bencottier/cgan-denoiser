# -*- coding: utf-8 -*-
"""
Finite measurement module for MRI data

Created on Tue Sep 27 13:57:13 2016

@author: uqscha22
"""
import radon
import farey #local module
from scipy import ndimage
import scipy.fftpack as fftpack
import pyfftw
import numpy as np
import math

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

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

def computeLines(kSpace, angles, centered = True, twoQuads = False):
    '''
    compute finite lines coordinates
    Returns a list or list of slice 2-tuples and corresponding list of m values
    '''
    p, s = kSpace.shape
    lines = []
    mValues = []
    for angle in angles:
        m, inv = farey.toFinite(angle, p)
        u, v = radon.getSliceCoordinates2(m, kSpace, centered, p)
        lines.append((u,v))
        mValues.append(m)
        #second quadrant
        if twoQuads:
            if m != 0 and m != p: #dont repeat these
                m = p-m
                u, v = radon.getSliceCoordinates2(m, kSpace, centered, p)
                lines.append((u,v))
                mValues.append(m)
    
    return lines, mValues

def computeKatzLines(kSpace, anglesSorted, finiteAnglesSorted, K, centered = True, twoQuads = False):
    '''
    compute finite lines coordinates given Katz criterion
    Returns a list or list of slice 2-tuples and corresponding list of angles and m values
    perp computes the perpendicular (s) lines as well
    '''
    N, M = kSpace.shape
    lines = []
    angles = []
    mValues = []
    for m, angle in zip(finiteAnglesSorted, anglesSorted):
        if isKatzCriterion(N, N, angles, K):
            print("Katz Criterion Met. Breaking")
            break
        m, inv = farey.toFinite(angle, N)
        u, v = radon.getSliceCoordinates2(m, kSpace, centered)
        lines.append((u,v))
        mValues.append(m)
        angles.append(angle)
        #second quadrant
        if twoQuads:
            if m != 0 and m != N: #dont repeat these
                m = N-m
                u, v = radon.getSliceCoordinates2(m, kSpace, centered)
                lines.append((u,v))
                mValues.append(m)
                p, q = farey.get_pq(angle)
                newAngle = farey.farey(-p,q) #not confirmed
                angles.append(newAngle)
    
    return lines, angles, mValues

def computeKatzLinesSubsets(s, kSpace, anglesSorted, finiteAnglesSorted, K, centered = True, twoQuads = False):
    '''
    compute finite lines coordinates given Katz criterion
    Returns a list or list of slice 2-tuples and corresponding list of angles and m values
    perp computes the perpendicular (s) lines as well
    '''
    N, M = kSpace.shape
    
    #create subsets
    subsetsLines = []
    subsetsAngles = []
    subsetsMValues = []
    for i in range(s):
        subsetsLines.append([])
        subsetsAngles.append([])
        subsetsMValues.append([])
    
    subsetIndex = 0 
    angles = []
    for m, angle in zip(finiteAnglesSorted, anglesSorted):
        if isKatzCriterion(N, N, angles, K):
            print("Katz Criterion Met. Breaking")
            break
        m, inv = farey.toFinite(angle, N)
        u, v = radon.getSliceCoordinates2(m, kSpace, centered)
        subsetsLines[subsetIndex].append((u,v))
        subsetsMValues[subsetIndex].append(m)
        subsetsAngles[subsetIndex].append(angle)
        angles.append(angle)
        #second quadrant
        if twoQuads:
            if m != 0 and m != N: #dont repeat these
                m = N-m
                u, v = radon.getSliceCoordinates2(m, kSpace, centered)
                subsetsLines[subsetIndex].append((u,v))
                subsetsMValues[subsetIndex].append(m)
                p, q = farey.get_pq(angle)
                newAngle = farey.farey(-p,q) #not confirmed
                subsetsAngles[subsetIndex].append(newAngle)
                angles.append(newAngle)
                
        subsetIndex += 1
        subsetIndex %= s
    
    return subsetsLines, subsetsAngles, subsetsMValues

def computePerpLines(kSpace, anglesSorted, finiteAnglesSorted, r, mValues, centered = True, twoQuads = False):
    '''
    compute finite lines coordinates given Katz criterion
    Returns a list or list of slice 2-tuples and corresponding list of angles and m values
    perp computes the perpendicular (s) lines as well
    '''
    N, M = kSpace.shape
    perpAngles = []
    linesPerp = []
    sValues = []
    for s in range(0, int(N/8*r)):
        m = N+s
        if m in mValues or m in sValues: #skip
            continue
        
        index = finiteAnglesSorted.index(m)
        angle = anglesSorted[index] #corresponding angle
        u, v = radon.getSliceCoordinates2(m, kSpace, centered)
        linesPerp.append((u,v))
        sValues.append(m)
        perpAngles.append(angle)
        if twoQuads:
            m = N-m
            u, v = radon.getSliceCoordinates2(m, kSpace, centered)
            linesPerp.append((u,v))
            sValues.append(m)
            perpAngles.append(angle)
    
    return linesPerp, perpAngles, sValues

#fractal creation
def finiteFractal(N, K, sortBy='Euclidean', twoQuads=True, centered=False):
    '''
    Create the finite fractal for image size N given the Katz criterion K
    
    sortBy can be 'Euclidean' for L2 norm or 'length' for L1 norm
    twoQuads can be used to cover the half plane
    
    Returns lines, angles, mValues, fractal formed (as an image), oversampling filter if applicable
    '''
    fareyVectors = farey.Farey()        
    fareyVectors.compactOn()
    fareyVectors.generateFiniteWithCoverage(N)
    
    #sort to reorder result for prettier printing
    finiteAnglesSorted, anglesSorted = fareyVectors.sort(sortBy)
    
    kSpace = np.zeros((N,N))
    lines, angles, mValues = computeKatzLines(kSpace, anglesSorted, finiteAnglesSorted, K, centered, twoQuads)
    mu = len(lines)
    print("Number of finite lines in fractal:", mu)
    
    samplesImage1 = np.zeros((N,N), np.float32)
    for line in lines:
        u, v = line
        for x, y in zip(u, v):
            samplesImage1[x, y] += 1
    #determine oversampling because of power of two size
    #this is fixed for choice of M and m values
    oversamplingFilter = np.zeros((N,N), np.uint32)
    onesSlice = np.ones(N, np.uint32)
    for m in mValues:
        radon.setSlice(m, oversamplingFilter, onesSlice, 2)
    oversamplingFilter[oversamplingFilter==0] = 1
    samplesImage1 /= oversamplingFilter
#    samplesImage1 = fftpack.fftshift(samplesImage1)
    
    return lines, angles, mValues, samplesImage1, oversamplingFilter
    
# Measure finite slices
def measureSlices(dftSpace, lines, mValues, dtype=np.float64):
    '''
    Measure finite slices of the DFT and convert to FRT projections
    Returns FRT projections
    '''
    N, M = dftSpace.shape
    mu = int(N+N/2)
    if N % 2 == 1: # if odd, assume prime
        mu = int(N+1)
    drtSpace = np.zeros((mu, M), dtype=dtype)
    for i, line in enumerate(lines):
        u, v = line
        sliceReal = ndimage.map_coordinates(np.real(dftSpace), [u,v])
        sliceImag = ndimage.map_coordinates(np.imag(dftSpace), [u,v])
        slice = sliceReal+1j*sliceImag
        finiteProjection = np.real(fftpack.ifft(slice)) # recover projection using slice theorem
        drtSpace[mValues[i],:] = finiteProjection
        
    return drtSpace
    
def measureSlices_complex(dftSpace, lines, mValues):
    '''
    Measure finite slices of the DFT and convert to FRT projections
    Returns FRT projections
    '''
    N, M = dftSpace.shape
    mu = int(N+N/2)
    if N % 2 == 1: # if odd, assume prime
        mu = int(N+1)
    drtSpace = np.zeros((mu, M), dtype=np.complex64)
    for i, line in enumerate(lines):
        u, v = line
        sliceReal = ndimage.map_coordinates(np.real(dftSpace), [u,v])
        sliceImag = ndimage.map_coordinates(np.imag(dftSpace), [u,v])
        slice = sliceReal+1j*sliceImag
        finiteProjection = fftpack.ifft(slice) # recover projection using slice theorem
        drtSpace[mValues[i],:] = finiteProjection
        
    return drtSpace

def frt(image, N, dtype=np.float32, mValues=None, center=False):
    '''
    Compute the DRT in O(n logn) complexity using the discrete Fourier slice theorem and the FFT.
    Input should be an NxN image to recover the DRT projections (bins), where N is prime.
    Float type is returned by default to ensure no round off issues.
    '''
    mu = N+1
    p = 0
    if N%2 == 0:
        mu = int(N+N/2)
        p = 2
        
    #FFT image
    fftLena = fftpack.fft2(image) #the '2' is important
    
    bins = np.zeros((mu,N),dtype=dtype)
    for m in range(0, mu):
        if mValues and m not in mValues:
            continue 
        
        slice = radon.getSlice(m, fftLena, center, p)
#        print slice
#        slice /= N #norm FFT
        projection = np.real(fftpack.ifft(slice))
        #Copy and norm
        for j in range(0, N):
            bins[m, j] = projection[j]
    
    return bins
    
def frt_complex(image, N, dtype=np.complex, mValues=None, center=False):
    '''
    Compute the DRT in O(n logn) complexity using the discrete Fourier slice theorem and the FFT.
    Input should be an NxN image to recover the DRT projections (bins), where N is prime.
    Float type is returned by default to ensure no round off issues.
    '''
    mu = N+1
    p = 0
    if N%2 == 0:
        mu = int(N+N/2)
        p = 2
        
    #FFT image
    fftLena = fftpack.fft2(image) #the '2' is important
    
    bins = np.zeros((mu,N),dtype=dtype)
    for m in range(0, mu):
        if mValues and m not in mValues:
            continue
        
        slice = radon.getSlice(m, fftLena, center, p)
#        print slice
#        slice /= N #norm FFT
        projection = fftpack.ifft(slice)
        if center:
            projection = fftpack.ifftshift(projection)
        #Copy and norm
        for j in range(0, N):
            bins[m, j] = projection[j]
    
    return bins

def ifrt(bins, N, norm = True, center = False, Isum = -1, mValues=None, oversampleFilter=None):
    '''
    Compute the inverse DRT in O(n logn) complexity using the discrete Fourier slice theorem and the FFT.
    Input should be DRT projections (bins) to recover an NxN image, where N is prime.
    projNumber is the number of non-zero projections in bins. This useful for backprojecting mu projections where mu < N.
    Isum is computed from first row if -1, otherwise provided value is used
    Finite angles not in mValues are ignored
    Coefficients are filtered using the exact oversampling filter if provided
    '''
    if Isum < 0:
        Isum = bins[0,:].sum()
#    print "ISUM:", Isum
    dftSpace = np.zeros((N,N),dtype=np.complex)
    
    p = 0
    if N%2 == 0:
        p = 2
    
    #Set slices (0 <= m <= N)
    for k, row in enumerate(bins): #iterate per row
        if mValues and k not in mValues:
            continue
        
        slice = fftpack.fft(row)
        radon.setSlice(k,dftSpace,slice,p)
    
    #exact filter (usually for non-prime sizes)
    if oversampleFilter is not None:
        dftSpace /= oversampleFilter
    else:
        dftSpace[0,0] -= float(Isum)*N

    #iFFT 2D image
    result = fftpack.ifft2(dftSpace)
    if not norm:
        result *= N #ifft2 already divides by N**2
    if center:
        result = fftpack.fftshift(result)

    return np.real(result)
    
def ifrt_complex(bins, N, norm = True, center = False, Isum = -1, mValues=None, oversampleFilter=None):
    '''
    Compute the inverse DRT in O(n logn) complexity using the discrete Fourier slice theorem and the FFT.
    Input should be DRT projections (bins) to recover an NxN image, where N is prime.
    projNumber is the number of non-zero projections in bins. This useful for backprojecting mu projections where mu < N.
    Isum is computed from first row if -1, otherwise provided value is used
    Finite angles not in mValues are ignored
    Coefficients are filtered using the exact oversampling filter if provided
    '''
    if Isum < 0:
        Isum = bins[0,:].sum()
#    print "ISUM:", Isum
    dftSpace = np.zeros((N,N),dtype=np.complex)
    
    p = 0
    if N%2 == 0:
        p = 2
    
    #Set slices (0 <= m <= N)
    for k, row in enumerate(bins): #iterate per row
        if mValues and k not in mValues:
            continue
        
        slice = fftpack.fft(row)
        radon.setSlice(k,dftSpace,slice,p)
    
    #exact filter (usually for non-prime sizes)
#    print("Oversampling Shape:", oversampleFilter.shape)
    if oversampleFilter is not None:
#        print("OVERSAMPLING!")
        dftSpace /= oversampleFilter
    else:
        dftSpace[0,0] -= float(Isum)*N

    #iFFT 2D image
    result = fftpack.ifft2(dftSpace)
    if not norm:
        result *= N #ifft2 already divides by N**2
    if center:
        result = fftpack.ifftshift(result)

    return result

def mse(img1, img2):
    '''
    Compute the MSE of two images using mask if given
    '''
    error = ((img1 - img2) ** 2).mean(axis=None)
    
    return error
    
def psnr(img1, img2, maxPixel=255):
    '''
    Compute the MSE of two images using mask if given
    '''
    error = mse(img1,img2)
    psnr_out = 20 * math.log(maxPixel / math.sqrt(error), 10)
    
    return psnr_out

#import random

def noise(kSpace, snr):
    '''
    Create noise in db for given kSpace and SNR
    '''
    r, s = kSpace.shape
    #pwoer of signal and noise
    P = np.sum(np.abs(kSpace)**2)/(r*s)
    P_N = P / (10**(snr/10))
    #P_N is equivalent to sigma**2 and signal usually within 3*sigma
    sigma = math.sqrt(P_N)
    
    noise = np.zeros_like(kSpace)
    for u, row in enumerate(kSpace):
        for v, coeff in enumerate(row):
            noiseReal = np.random.normal(0, sigma)
            noiseImag = np.random.normal(0, sigma)
            noise[u,v] = noiseReal + 1j*noiseImag
            
    return noise
