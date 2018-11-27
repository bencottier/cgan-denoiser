'''
Read Nifti 3D volumes via NiBabel and extract slices from 3D volume then save them as nifti files
'''
#get list of images
import filenames
#load modules for arrays and nifti file support
import numpy as np
import nibabel as nib
#import scipy.misc

path = "disc10/" #3D volumes
caseIndex = 1
sliceView = 2
offsetView = 5
offsets = [-2, -1, 0, 1, 2]

img_path = "PROCESSED/MPRAGE/T88_111/"
outpath = "slices/"
output_prefix = "case_"
#img_path = "FSL_SEG/"
#outpath = "slices_seg/"
#output_prefix = "seg_"

dirList, caseList = filenames.getSortedFileListAndCases(path, caseIndex, '*', True)

#process each 3D volume
for dir, case in zip(dirList, caseList):
    #Image
    #get list of filenames and case IDs from path where 3D volumes are
    imageList, caseList = filenames.getSortedFileListAndCases(dir+'/'+img_path, caseIndex, "*.hdr")
#    print(imageList)
#    print(caseList)
    
    if not imageList:
        print("Image Not Found. Skipping", case)
        continue

    #load nifti file
    image = imageList[0]
    img = nib.load(dir+'/'+img_path+image)
    print("Loaded", image)

    #get the numpy array version of the image
    data = img.get_data() #numpy array without orientation
    print("Image shape:", data.shape)
#    print("Image type:", data.dtype)

    count = 0
    for offset in offsets:
        #extract a slice from 3D volume to save
        if sliceView == 1: 
            img_slice = data[:,int(data.shape[1]/2)+offsetView+offset,:] #coronal view, slice middle of x-z plane
        elif sliceView == 0:
            img_slice = data[int(data.shape[0]/2)+offsetView+offset,:,:] #coronal view, slice middle of x-z plane
        else:
            img_slice = data[:,:,int(data.shape[2]/2)+offsetView+offset] #coronal view, slice middle of x-z plane
        #~ img_slice = np.fliplr(img_slice) #flipped x-axis when reading
        img_slice = np.flipud(img_slice) #flipped x-axis when reading
#        print("Slice shape:", img_slice.shape)
    
        #save slice
        slice = nib.Nifti1Image(img_slice, np.eye(4))
        outname = outpath + output_prefix + str(case).zfill(3) + "_slice_" + str(count) + ".nii.gz"
        slice.to_filename(outname)
        count += 1
    
#    break
