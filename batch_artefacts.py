# -*- coding: utf-8 -*-
"""
Process 2D slices and produce turbulent artefacts

Created on Wed Nov 21 10:28:15 2018

@author: uqscha22
"""
#get list of images
import filenames
#load modules for arrays and nifti file support
import numpy as np
import nibabel as nib
import finite
import scipy.fftpack as fftpack
import pyfftw

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

N = 256
K = 2.4
path = "slices/" #3D volumes
path_seg = "slices_seg/" #3D volumes
outpath = "slices_artefact/"
#outpath = "slices_artefact_rand/"
outpath2 = "slices_pad/"
outpath3 = "slices_seg_pad/"
output_prefix = "case_"
output_prefix2 = "seg_"
caseIndex = 0

#setup fractal
lines, angles, mValues, fractal, oversampling = finite.finiteFractal(N, K, sortBy='Euclidean', twoQuads=True)
mu = len(lines)
print("Number of finite lines:", mu)
print("Number of finite points:", mu*(N-1))

imageList, caseList = filenames.getSortedFileListAndCases(path, caseIndex, "*.nii.gz", True)
imageList, sliceList = filenames.getSortedFileListAndCases(path, caseIndex+1, "*.nii.gz", True)
#print(imageList)
#print(caseList)

segList, segCaseList = filenames.getSortedFileListAndCases(path_seg, caseIndex, "*.nii.gz", True)

#process each 3D volume
count = 0
for image, seg, case, sliceIndex in zip(imageList, segList, caseList, sliceList):
    img = nib.load(image)
    lbl = nib.load(seg)
    print("Loaded", image)

    #get the numpy array version of the image
    data = img.get_data() #numpy array without orientation
    lbl_data = lbl.get_data() #numpy array without orientation
    lx, ly, lz = data.shape
    print("Image shape:", data.shape)
    
    #pad
    mid = int(N/2.0)
    midx = int(lx/2.0+0.5)
    midy = int(ly/2.0+0.5)
    newLengthX1 = mid - midx
    newLengthX2 = mid + midx
    newLengthY1 = mid - midy
    newLengthY2 = mid + midy
    newImage = np.zeros((N,N))
    newLabels = np.zeros((N,N))
#    imageio.imcrop(data, N, m=0, center=True, out_dtype=np.uint32)
    newImage[newLengthX1:newLengthX2, newLengthY1:newLengthY2] = data[:,:,0]
    newLabels[newLengthX1:newLengthX2, newLengthY1:newLengthY2] = lbl_data[:,:,0]
    
    #save padded image
    slicePadded = nib.Nifti1Image(newImage, np.eye(4))
    outname = outpath2 + output_prefix + str(case).zfill(3) + "_slice_" + str(sliceIndex) + ".nii.gz"
    slicePadded.to_filename(outname)
    segPadded = nib.Nifti1Image(newLabels, np.eye(4))
    outname = outpath3 + output_prefix2 + str(case).zfill(3) + "_slice_" + str(sliceIndex) + ".nii.gz"
    segPadded.to_filename(outname)
    
    #2D FFT
    kSpace = fftpack.fft2(newImage) #the '2' is important
#    fftkSpaceShifted = fftpack.fftshift(kSpace)
    kSpace *= fractal
    artefactImage = fftpack.ifft2(kSpace) #the '2' is important
    artefactImage = np.real(artefactImage)
    
    slice = nib.Nifti1Image(artefactImage, np.eye(4))
    outname = outpath + output_prefix + str(case).zfill(3) + "_slice_" + str(sliceIndex) + ".nii.gz"
    slice.to_filename(outname)
    count += 1
    
#    break

fractalImg = nib.Nifti1Image(fractal, np.eye(4))
outname = outpath + "fractal.nii.gz"
fractalImg.to_filename(outname)

np.savez('artefact_arrays.npz', lines=lines, angles=angles, mValues=mValues, fractal=fractal, oversampling=oversampling)
    
print("Total", count, "processed")
    