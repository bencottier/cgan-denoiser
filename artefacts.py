# -*- coding: utf-8 -*-
"""
artefacts.py

Functions to artificially induce artefacts in images. 

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function
from data_processing import normalise
from finitetransform import farey, radon
import finite
import numpy as np
from scipy import fftpack
from scipy.stats import norm
import math


def add_gaussian_noise(data, stdev=0.1, mean=0.0, data_range=(0, 1), clip=True):
    """
    Add noise to array data, sampled from a normal/Gaussian distribution.

    Assumes the data is limited to the range [-1.0, 1.0].

    Arguments:
        data: ndarray. Data to add noise to.
        stdev: float > 0. Standard deviation of the noise distribution.
        mean: float. Mean (average) of the noise distribution.
        clip: bool. If True, limit the resulting data to [-1.0, 1.0]
    """
    data_ = normalise(data, (-1, 1), data_range)
    noisy = data_ + np.random.normal(mean, stdev, data.shape)
    noisy = np.clip(noisy, -1, 1) if clip else noisy
    return normalise(noisy, data_range, (-1, 1))


def add_space_noise(data, fwhm=1.4, sig=1.2, data_range=(0, 1)):
    """
    ** Requires OpenCV-Python (`cv2`) **
    Add noise to array data, approximating degradation in astronomical 
    telescope imaging. Data must be at least 3D, of shape
    (n, rows, columns, [channel])

    Based on [Schawinski et al., 2017](https://arxiv.org/abs/1702.00403)
    ([pdf](https://arxiv.org/pdf/1702.00403.pdf), 
    [code](https://github.com/SpaceML/GalaxyGAN/blob/master/roou.m)).
    
    Real sources of this degradation include the sensing element,
    the sky background, angular resolution, and atmospheric distortion.
    The approximate model used here convolves the data with a Gaussian 
    point spread function (PSF) and adds Gaussian white noise.

    Arguments:
        data: ndarray. Data to add noise to.
        fwhm: float > 0. Full-width at half maximum of the Gaussian PSF
            in arcseconds.
        sig: float > 0. White noise level.

    Returns:
        The modified data, normalised then stretched with arcsinh.
    """
    import cv2 as cv
    # Constants
    # Conversion to standard deviation: 2*sqrt(2*ln(2)) * stdev ~= 2.355 * stdev
    # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    FWHM_TO_STDEV = 2.355
    # Pixel scale in the Sloan Digital Sky Survey
    # Slide 3: http://hosting.astro.cornell.edu/academics/courses/astro7620/docs/sdss_shan.pdf
    ARCSEC_TO_PX = 0.396
    ks = 5  # filter kernel size
    nt = 0.1  # noise threshold
    vmin = -0.1  # data value cutoff
    vmax =  4.0  # data value cutoff
    growth_rate = 10.0  # arcsinh stretch factor
    scaling = 3.0  # arcsinh stretch factor
    gaussian_stdev = (fwhm / ARCSEC_TO_PX) / FWHM_TO_STDEV

    if len(data.shape) == 1:
        raise ValueError("Data must be at least 2D")
    data_temp = normalise(data, (-1, 1), data_range).astype('float32')

    for i in range(data_temp.shape[0]):
        x = data_temp[i]
        # Add gaussian blur
        x_blur = cv.GaussianBlur(x, (ks, ks), gaussian_stdev)
        # Estimate the distribution of low-power data
        x_near_zero = x[x > -nt]
        x_near_zero = x_near_zero[x_near_zero < nt]
        m1, s1 = norm.fit(x_near_zero)
        x_blur_near_zero = x_blur[x_blur > -nt]
        x_blur_near_zero = x_blur_near_zero[x_blur_near_zero < nt]
        m2, s2 = norm.fit(x_blur_near_zero)

        # Add noise
        # VAR(X + Y) = VAR(X) + VAR(Y) if X, Y are independent
        noise_var = (sig * s1)**2 - s2**2
        noise_var = 1e-8 if noise_var <= 0 else noise_var
        noise = np.random.normal(0.0, math.sqrt(noise_var), x.shape[:2])
        x_degraded = np.zeros_like(x)
        if len(data_temp.shape) >= 4:
            for c in range(data_temp.shape[-1]):
                x_degraded[..., c] = x_blur[..., c] + noise
        else:
            x_degraded = x_blur + noise

        # Clip
        x_degraded = np.clip(x_degraded, vmin, vmax)
        # Normalise
        x_degraded = (x_degraded - vmin) / (vmax - vmin)
        # arcsinh stretch
        x_degraded = np.arcsinh(growth_rate * x_degraded) / scaling

        data_temp[i] = x_degraded

    return data_temp.astype(data.dtype)


class Sampler(object):

    def __init__(self, seed=0):
        self.seed = seed
        self.mask = None
        self.r_no_tile = None
        self.r_actual = None

    def generate_mask(self, size):
        raise NotImplementedError("Method must be defined in subclass.")

    def sample_kspace(self, image):
        if self.mask is None:
            self.generate_mask(image.shape[:-2])
        kspace_input = fftpack.fft2(image.astype(np.complex64))
        kspace_sampled = kspace_input * self.mask
        image_sampled = fftpack.ifft2(kspace_sampled)
        image_sampled_real = np.real(image_sampled)[..., np.newaxis]
        image_sampled_imag = np.imag(image_sampled)[..., np.newaxis]
        image_sampled_cat = np.concatenate([image_sampled_real, image_sampled_imag], axis=2)
        return image_sampled_cat.astype(np.float32)

    def do_transform(self, x):
        x = self.sample_kspace(x)
        return x


class OneDimCartesianRandomSampler(Sampler):
    """
    Transform the image to Fourier space, sample the array
    along randomly-spaced axis-aligned lines, then convert
    the result back to image space.
    
    Terminology comes from the MRI domain, where this method has relevance.
    """
    
    def __init__(self, r=5.0, r_alpha=3, axis=1, acs=3, seed=0):
        """
        Arguments:
            r: float. Reduction factor. A 1/r fraction of k-space will be sampled,
                excluding the auto-calibration signal region.
            r_alpha: float. Higher values mean lower-frequency samples are more likely.
            axis: int. The axis the sampling lines are aligned to.
            acs: int. Width of the auto-calibration signal region, 
                which is fully-sampled about the center of k-space.
            seed: int. Random seed. A positive number gives repeatable "randomness".
        """
        super(OneDimCartesianRandomSampler, self).__init__(seed)
        self.r,self.r_alpha,self.axis,self.acs = r,r_alpha,axis,acs

        
    def generate_mask(self, size):
        """
        Generates a mask array for variable-density cartesian 1D sampling.
        Sampling probability is proportional to the position along the sampling axis,
        such that lower frequencies are more likely. The alpha value controls the 
        proportionality, e.g. alpha = 1 is linear, alpha = 2 is square.
        """
        # Initialise
        if self.seed >= 0:
            np.random.seed(self.seed)
        if type(size) != tuple or len(size) != 2:
            raise ValueError("Size must be a 2-tuple of ints")
        mask = np.zeros(size)
        # Get sample coordinates
        num_phase_encode = size[self.axis]
        num_phase_sampled = int(np.floor(num_phase_encode / self.r))
        coordinate_normalized = np.arange(num_phase_encode)
        coordinate_normalized = np.abs(coordinate_normalized - num_phase_encode/2) \
                                / (num_phase_encode/2.0)
        prob_sample = coordinate_normalized**self.r_alpha
        prob_sample = prob_sample / np.sum(prob_sample)
        index_sample = np.random.choice(num_phase_encode, size=num_phase_sampled, 
                                        replace=False, p=prob_sample)
        # Set the samples in the mask
        if self.axis == 0:
            mask[index_sample, :] = 1
        else:
            mask[:, index_sample] = 1
        self.r_no_tile = len(mask.flatten()) / np.sum(mask.flatten())
        # ACS
        acs1 = int((self.acs + 1) / 2)
        acs2 = -int(self.acs / 2)
        if self.axis == 0:
            mask[:acs1, :] = 1
            mask[acs2:, :] = 1
        else:
            mask[:, :acs1] = 1
            mask[:, acs2:] = 1
        # Compute reduction
        self.r_actual = len(mask.flatten()) / np.sum(mask.flatten())
        self.mask = mask
        return mask
    

class FractalRandomSampler(Sampler):
    """
    Transform the image to Fourier space, sample the array
    in a partially random fractal pattern composed of angled lines, 
    then convert the result back to image space.
    
    Terminology comes from the MRI domain, where this method has relevance.
    """
    
    def __init__(self, k=1, K=0.1, r=0.48, ctr=1/8, two_quads=True, center=False, seed=0):
        """
        Arguments:
            k: 
            K: float. Relates to the Katz criterion.
                Indirectly controls the reduction factor.
            r: float.
            ctr: float. Centre tiling radius as a fraction of the image height.
            two_quads: bool. If True, generate separate patterns in two quadrants
                instead of one.
            seed: int. Random seed. Non-negative int for repeatable pseudo-randomness.
        """
        super(FractalRandomSampler, self).__init__(seed)
        self.k,self.K,self.r,self.ctr,self.two_quads,self.center,self.seed = k,K,r,ctr,two_quads,center,seed
        
    def generate_mask(self, size):
        """
        Generates a sampling_mask array for fractal sampling.
        The fractal is derived from Farey vectors and composed of straight
        lines at varied angles.
        Some lines may be randomly configured to introduce incoherence in 
        the resulting artefacts.
        """
        # Initialise
        seed = self.seed if self.seed >= 0 else None
        np.random.seed(seed)
        # Generate lines in fractal
        N = size[0]
        M = self.k * N
        fareyVectors = farey.Farey()        
        fareyVectors.compactOn()
        fareyVectors.generateFiniteWithCoverage(N)
        finiteAnglesSorted, anglesSorted = fareyVectors.sort('length')
        powSpect = np.zeros((M, M), dtype=np.float64)
        lines, angles, mValues = finite.computeRandomLines(
            powSpect, anglesSorted, finiteAnglesSorted, 
            self.r, self.K, centered=False, twoQuads=self.two_quads)
        # Set the samples in the sampling_mask from along the lines
        sampling_mask = np.zeros((M,M), np.float)
        for line in lines:
            u, v = line
            for x, y in zip(u, v):
                sampling_mask[x, y] += 1
        # Determine oversampling because of power of two size
        # This is fixed for choice of M and m values
        oversamplingFilter = np.zeros((M,M), np.float)
        onesSlice = np.ones(M, np.float)
        for m in mValues:
            radon.setSlice(m, oversamplingFilter, onesSlice, 2)
        oversamplingFilter[oversamplingFilter==0] = 1
        sampling_mask /= oversamplingFilter
        sampling_mask = fftpack.fftshift(sampling_mask)
        self.r_no_tile = len(sampling_mask.flatten()) / np.sum(sampling_mask.flatten())
        # Tile center region further
        radius = self.ctr * N
        centerX = M/2
        centerY = M/2
        count = 0
        for i, row in enumerate(sampling_mask):
            for j, col in enumerate(row):
                distance = math.sqrt( (i-float(centerX))**2 + (j-float(centerY))**2)
                if distance < radius:
                    if not sampling_mask[i, j] > 0: #already selected
                        count += 1
                        sampling_mask[i, j] = 1
        # Compute reduction
        if self.center:
            sampling_mask = fftpack.ifftshift(sampling_mask)
        self.r_actual = len(sampling_mask.flatten()) / np.sum(sampling_mask.flatten())
        self.mask = sampling_mask
        return sampling_mask.astype(np.uint32)


if __name__ == "__main__":
    from scipy.misc import ascent
    import tensorflow as tf
    import matplotlib.pyplot as plt

    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    # data = ascent()
    data = train_images
    print("Data shape: ", data.shape)
    data = add_gaussian_noise(normalise(data, (-1, 1), (0, 255)), 0.2)
    # plt.imshow(data[0], cmap='gray')

    fig = plt.figure(figsize=(5,5))

    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(data[i], cmap='gray')
        plt.axis('off')

    plt.show()

    data = ascent()
    data = 5.0 * (data - data.min()) / (data.max() - data.min()) + -0.2
    data_degraded = add_space_noise(data, 1.8, 5.0)

    plt.figure()
    plt.gray()
    plt.subplot(1, 2, 1)
    plt.imshow(data)
    plt.subplot(1, 2, 2)
    plt.imshow(data_degraded)
    plt.show()
