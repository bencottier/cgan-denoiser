'''
 Radon Transforms module

'''
import numpy as np
import numbertheory as nt#local modules
import farey
import random

#-------------
# project arrays
def drt(image, p, dtype=np.int32):
    '''
    Compute the Finite/Discrete Radon Transform of Grigoryan, Gertner, Fill and others using a polynomial time method.
    Returns the bins 'image'.
    Method is integer-only and doesn't need floats. Only uses additions and supports prime sizes.
    '''
    y = index = row = 0
    bins = np.zeros( (p+1, p), dtype )
    
    cimage = image.ravel() #get c-style array of image
    cbins = bins.ravel() #get c-style array of image
    
    for m in range(p):
        y = 0
        for x in range(p):
            row = x*p
            for t in range(p):
                index = row + (t + y)%p #Compute Column (in 1D)
                cbins[m*p+t] += cimage[index]
            y += m # Next pixel
            
    for t in range(p): #columns
        for y in range(p): #translates
            cbins[p*p+t] += cimage[t*p+y]
    
    return bins
    
def drt_dyadic(image, N):
    '''
    Compute the Finite/Discrete Radon Transform of Hsung et al. using a polynomial time method.
    Returns the bins 'image'.
    Method is integer-only and doesn't need floats. Only uses additions and supports dyadic sizes.
    '''
    y = index = row = 0
    bins = np.zeros( (N+int(N/2), N) )
    
    cimage = image.ravel() #get c-style array of image
    cbins = bins.ravel() #get c-style array of image
    
    for m in range(N):
        y = 0
        for x in range(N):
            row = x*N
            for t in range(N):
                index = row + (t + y)%N #Compute Column (in 1D)
                cbins[m*N+t] += cimage[index]
            y += m # Next pixel
            
    for s in range(0, int(N/2)): #Perp Projection Angle
        x = 0
        for y in range(N): #translates
            for t in range(N): #columns            
                index = ( (x+t)%N )*N + y #Compute Row (in 1D)
                cbins[(s+N)*N+t] += cimage[index]
            x += 2*s # Next pixel
    
    return bins
    
def idrt(bins, p, norm = True):
    '''
    Compute the inverse Finite/Discrete Radon Transform of Grigoryan, Gertner, Fill and others using a polynomial time method.
    Returns the recovered image.
    Method is exact and integer-only and doesn't need floats
    '''
    y = index = row = 0
    image = np.zeros( (p, p) )
    
    cimage = image.ravel() #get c-style array of image
    cbins = bins.ravel() #get c-style array of image
    
    Isum = 0
    for t in range(p):
        Isum += cbins[t]
        
    for m in range(p):
        y = 0
        for x in range(p):
            row = x*p
            for t in range(p):
                index = row + (t + y)%p #Compute Column (in 1D)
                cimage[m*p+t] += cbins[index]
            y += p-m # Next pixel
            
    for t in range(p): #columns
        for y in range(p): #translates
            cimage[t*p+y] += cbins[p*p+t]
            cimage[t*p+y] -= Isum
            if norm:
                cimage[t*p+y] /= p
    
    return image

#-------------
#fast versions with FFTs and NTTs
import scipy.fftpack as fftpack
import pyfftw

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

def frt(image, N, dtype=np.float64, center=False):
    '''
    Compute the DRT in O(n logn) complexity using the discrete Fourier slice theorem and the FFT.
    Input should be an NxN image to recover the DRT projections (bins).
    Float type is returned by default to ensure no round off issues.
    Assumes object is real valued
    '''
    p = 2
    mu = int(N+N/2)
    if N % 2 == 1: # if odd, assume prime
        mu = int(N+1)
        p = 0
        
    #FFT image
    fftLena = fftpack.fft2(image) #the '2' is important
    
    bins = np.zeros((mu,N),dtype=dtype)
    for m in range(0, mu):
        slice = getSlice(m, fftLena, center, p)
#        print slice
#        slice /= N #norm FFT
        projection = np.real(fftpack.ifft(slice))
        #Copy and norm
        for j in range(0, N):
            bins[m, j] = projection[j]
#        print projection - bins[m, :]
    
    return bins
        
def ifrt(bins, N, norm = True, center = False, Isum = -1):
    '''
    Compute the inverse DRT in O(n logn) complexity using the discrete Fourier slice theorem and the FFT.
    Input should be DRT projections (bins) to recover an NxN image.
    projNumber is the number of non-zero projections in bins. This useful for backprojecting mu projections where mu < N.
    Isum is computed from first row if -1, otherwise provided value is used
    '''
    if Isum < 0:
        Isum = bins[0,:].sum()
#    print "ISUM:", Isum

    result = np.zeros((N,N),dtype=np.complex64)
    filter = oversampling_1D_filter(N,2,norm) #fix DC for dyadic
    
    p = 2
    if N % 2 == 1: # if odd, assume prime
        filter = np.ones(N)
#        filter[0] = 1.0/(projNumber+1) #DC fix
        filter[0] = 1.0
        p = 0
    
    #Set slices (0 <= m <= N)
    for k, row in enumerate(bins): #iterate per row
        slice = fftpack.fft(row)
        slice *= filter
#        print "m:",k
        setSlice(k,result,slice,p)
#    print "filter:", filter
    
    if N % 2 == 1: # if odd, assume prime
        result[0,0] -= float(Isum)*N

    #iFFT 2D image
    result = fftpack.ifft2(result)
    if not norm:
        result *= N #ifft2 already divides by N**2
    if center:
        result = fftpack.fftshift(result)

    return np.real(result)

#-------------
#helper functions
def getSlice(m, data, center=False, p=0):
    '''
    Get the slice (at m) of a NxN discrete array using the discrete Fourier slice theorem.
    This can be applied to the DFT or the NTT arrays.
    '''
    rows, cols = data.shape
    if p == 0:
        p = rows #base of image size, prime size this is p

    slice = np.zeros(rows, dtype=data.dtype)
    if m < cols and m >= 0:
#        for k, col in enumerate(data.T): #iterate per column, transpose is cheap
        for k in range(0,cols):
            index = (rows-(k*m)%rows)%rows
            slice[k] = data[index,k]
    else: #perp slice, assume dyadic
        s = m-cols #consistent notation
#        for k, row in enumerate(data): #iterate per row
        for k in range(0,rows):
            index = (cols-(k*p*s)%cols)%cols
            slice[k] = data[k,index]
            
    if center:
        sliceCentered = np.copy(slice)
        offset = int(rows/2.0)
        for index, value in enumerate(sliceCentered):
            newIndex = (index+offset)%rows
            slice[newIndex] = value

    return slice
    
def getSliceCoordinates(m, data, center=False, b=0):
    '''
    Get the slice coordinates u, v arrays (in pixel coordinates at finite angle m) of a NxN discrete array using the discrete Fourier slice theorem.
    This can be applied to the DFT or the NTT arrays and is good for drawing sample points of the slice on plots.
    b can be used to stop the slice earlier, say sampling b<p.
    '''
    rows, cols = data.shape
    p = rows #base of image size, prime size this is p
    B = (b-1)/2
    
    offset = 0
    if center:
        offset = int(rows/2.0)
#    print "offset:", offset

    u = []
    v = []
    if m < cols and m >= 0:
        extentMax = (rows-(B*m)%rows + offset)%rows
        extentMin = (rows-(B*m)%rows - offset)%rows
#        print "extentMax:", extentMax, "extentMin:", extentMin
        for k, col in enumerate(data.T): #iterate per column, transpose is cheap
            x = (k+offset)%rows
            index = (rows-(k*m)%rows + offset)%rows
            if b > 0 and (index > extentMax or index < extentMin):
                continue
            v.append(x)
            u.append(index)
    else: #perp slice, assume dyadic
        s = m #consistent notation
        extentMax = ((cols-B*2*s)%cols + cols + offset)%cols
        extentMin = ((cols-B*2*s)%cols + cols - offset)%cols
#        print "extentMax:", extentMax, "extentMin:", extentMin
        for k, row in enumerate(data): #iterate per row
            x = (k+offset)%cols
            index = ((cols-k*p*s)%cols + cols + offset)%cols
            if b > 0 and (index > extentMax or index < extentMin):
                continue
            v.append(x)
            u.append(index)

    return np.array(u), np.array(v)
    
def getSliceCoordinates2(m, data, center=False, p=2):
    '''
    Get the slice coordinates u, v arrays (in pixel coordinates at finite angle m) of a NxN discrete array using the discrete Fourier slice theorem.
    This can be applied to the DFT or the NTT arrays and is good for drawing sample points of the slice on plots.
    '''
    rows, cols = data.shape
    
    offset = 0
    if center:
        offset = int(rows/2.0)
#    print "offset:", offset

    u = []
    v = []
    if m < cols and m >= 0:
        for k, col in enumerate(data.T): #iterate per column, transpose is cheap
            x = (k+offset)%rows
            index = (rows-(k*m)%rows + offset)%rows
#            print "k:",k,"\tindex:",index
            v.append(x)
            u.append(index)
    else: #perp slice, assume dyadic
        s = m-cols #consistent notation
#        print "s:",s
        for k, row in enumerate(data): #iterate per row
            x = (k+offset)%cols
            index = (cols-(k*p*s)%cols + offset)%cols
#            print "k:",k,"\tindex:",index
            u.append(x)
            v.append(index)

    return np.array(u), np.array(v)

def getMojetteProjectionLength(angle, P, Q):
    '''
    Return the number of bins for projection at angle of a PxQ image.
    Wraps function from Farey module
    '''
    return farey.projectionLength(angle, P, Q) #no. of bins

def getMojetteSlice(angle, P, Q, data, center=False):
    '''
    Get the Mojette slice (at angle) of a NxN discrete array using the discrete non-periodic Fourier slice theorem.
    This can be applied to the DFT or the NTT arrays.
    '''
    N, N = data.shape
    p, q = farey.get_pq(angle)
    B = getMojetteProjectionLength(angle, P, Q)
#    print "B Mojette:", B
    offset = 0.0
    if center:
        offset = N/2.0

    slice = np.zeros(2*B-1, dtype=data.dtype)
#    slice = np.zeros(B, dtype=data.dtype)
    for translate in range(0, B):
        translateP = (p*translate)%N #has issues in C, may need checking
        translatePConjugate = (N-translateP+offset)%N #has issues in C, may need checking
        translateQ = (q*translate)%N #has issues in C, may need checking
        translateQConjugate = (N-translateQ+offset)%N #has issues in C, may need checking
        slice[translate] = data[(translateQ+offset)%N, (translateP+offset)%N]
        if translate != 0:
            slice[(2*B-1-translate)%N] = data[translateQConjugate, translatePConjugate]
#            slice[(B-translate)%N] = data[translateQConjugate, translatePConjugate]

    return slice

def getProjectionCoordinates(m, data, center=False, p=2):
    '''
    Get the projection coordinates x, y arrays (in pixel coordinates at finite angle m) of a NxN discrete array using the discrete Radon transform.
    This can be applied to the image arrays and is good for drawing sample points of the slice on plots.
    '''
    rows, cols = data.shape
    
    offset = 0
    if center:
        offset = int(rows/2)
#    print "offset:", offset

    x = []
    y = []
    if m < cols and m >= 0:
        for k, col in enumerate(data.T): #iterate per column, transpose is cheap
            u = (k+rows-offset)%rows
            index = (k%rows*m + cols - offset)%cols
#            print "u:", u, "v:", index
            x.append(u)
            y.append(index)
    else: #perp slice, assume dyadic
        s = m #consistent notation
        for k, row in enumerate(data): #iterate per row
            v = (k+cols-offset)%cols
            index = (k*p*s + rows - offset)%rows
            y.append(v)
            x.append(index)

    return np.array(x), np.array(y)
    
def mojetteProjection(projection, N, angle, P, Q):
    '''
    Convert a finite projection into a Mojette projection for given (p,q).
    
    Assumes you have correctly determined (p,q) for the m value of the finite projection.
    '''
#    dyadic = True
#    if N % 2 == 1: # if odd, assume prime
#        dyadic = False
        
    p, q = farey.get_pq(angle)
    B = farey.projectionLength(angle, P, Q) #no. of bins
    m, inv = farey.toFinite(angle, N)
    translateOffset, perp = farey.finiteTranslateOffset(angle, N, P, Q)
    mojetteProj = np.zeros(B)
    angleSign = p*q 
    
    '''if nt.is_coprime(q, N):
        inv = q
    else: #perp projection
        inv = p
    
    for translate, bin in enumerate(projection):
        translateMojette = int((inv*translate)%N)
        if angleSign >= 0 and perp: #Reverse for perp
            translateMojette = int(translateOffset) - translateMojette
        else:
            translateMojette += int(translateOffset)
        print "TR:", translate, "Tm:", translateMojette
        mojetteProj[translateMojette] += bin'''
        
    for translate, bin in enumerate(mojetteProj):
        if angleSign >= 0 and perp: #Reverse for perp
            translateMojette = int(translateOffset) - int(translate)
        else:
            translateMojette = int(translate) - int(translateOffset)
            
        if translateMojette < 0:
            translateFinite = ( N - ( inv*abs(translateMojette) )%N )%N
        else:         
            translateFinite = (inv*translateMojette)%N #has issues in C, may need checking
        mojetteProj[translate] += projection[translateFinite]
#        print "TR:", translateFinite, "TM:", translate
        
    return mojetteProj
    
def setSlice(m, data, slice, p=0):
    '''
    Set the slice (at m) of a NxN discrete array using the discrete Fourier slice theorem.
    This can be applied to the DFT or the NTT arrays.
    '''
    rows, cols = data.shape
    if p == 0:
        p = rows #base of image size, prime size this is p
    
    if m < cols and m >= 0:
#        for k, col in enumerate(data.T): #iterate per column, transpose is cheap
        for k in range(0,cols):
            index = (rows-(k*m)%rows)%rows
#            print "k:",k,"\tindex:",index
            data[index,k] += slice[k]
    else: #perp slice, assume dyadic
        s = m-cols #consistent notation
#        print "s:",s
#        for k, row in enumerate(data): #iterate per row
        for k in range(0,rows):
            index = (cols-(k*p*s)%cols)%cols
#            print "k:",k,"\tindex:",index
            data[k,index] += slice[k]

def setSlice_Integer(m, data, slice, modulus):
    '''
    Set the slice (at m) of a NxN discrete array using the discrete Fourier slice theorem.
    This can be applied to the DFT or the NTT arrays.
    '''
    rows, cols = data.shape
    p = rows #base of image size, prime size this is p
    
    if m < cols and m >= 0:
        for k, col in enumerate(data.T): #iterate per column, transpose is cheap
            index = (rows-(k*m)%rows)%rows
#            print "k:",k,"\tindex:",index
            col[index] += slice[k]
            col[index] %= modulus
    else: #perp slice, assume dyadic
        s = m-cols #consistent notation
#        print "s:",s
        for k, row in enumerate(data): #iterate per row
            index = (cols-(k*p*s)%cols)%cols
#            print "k:",k,"\tindex:",index
            row[index] += slice[k]
            row[index] %= modulus
    
def cyclicShift(image, p, fromRow, rotation):
    '''
    Cyclic shift a row from a 2D image/matrix by rotation amount.
    The row will wrap around the columns.
    Returns the shifted row.
    '''
    nextStep = 0
    rot = (rotation + p)%p #handle negative values
    
    rotRow = np.zeros(p)
    cimage = image.ravel() #get c-style array of image
    #~ print "Size:", cimage.shape
    
    for t in range(p): #columns
        nextStep = (t + rot)%p
        #~ rotRow[t] = cimage[nextStep + fromRow*p] #unshift
        rotRow[nextStep] = cimage[t + fromRow*p]
    
    return rotRow
    
def cyclicShiftSlice(slice, p, rotation):
    '''
    Cyclic shift a slice by rotation amount.
    The slice will wrap around the columns.
    Returns the shifted slice.
    '''
    rot = (rotation + p)%p #handle negative values
    rotFilter = np.zeros(p)
    
    rotFilter[rot] = 1
    rotSlice = fftpack.fft(rotFilter)*slice
    
    return rotSlice
    
def cyclicShuffle(image, p, translate, fromRow, shuffleStep):
    '''
    Cyclically shuffle the column values by step with wrap around.
    This results in a interlacing of the signal.
    '''
    index = 0
    step = (shuffleStep+p)%p
  
    shuffleRow = np.zeros(p)
    cimage = image.ravel() #get c-style array of image
    #~ print "Size:", cimage.shape
    
    for x in range(p):
        index = (translate + x*step)%p
        if index < 0:
            index += p
        shuffleRow[(x + translate + p)%p] = cimage[fromRow*p + index] #unshuffle  
        
    return shuffleRow
    
def cyclicUnshuffle(image, p, translate, fromRow, shuffleStep):
    '''
    Cyclically unshuffle the column values by step with wrap around.
    This results in a undoing of a interlacing of the signal, if the step matches the shuffle step.
    '''
    index = 0
    step = (shuffleStep+p)%p
  
    shuffleRow = np.zeros(p)
    cimage = image.ravel() #get c-style array of image
    #~ print "Size:", cimage.shape
    
    for x in range(p):
        index = (translate + x*step)%p
        if index < 0:
            index += p
        shuffleRow[index] = cimage[fromRow*p + (x + translate + p)%p] #unshuffle  
        
    return shuffleRow
    
#-------------
#Utility functions
def remapProjections(p, rotation):
    '''
    For a given p size, a Farey set is generated and rotated by rotation vector provided.
    The new mapping is returned, with their corresponding new Farey vectors.
    '''
    original_angles = []
    rotated_angles = []
    ms = []
    inverses = []
    m_dashs = []
    for m in range(p): #add the Farey angles corresponding to m values
        ms.append(m)
        #~ angle = complex(1.0, m)
        angle = complex(m, 1)
        original_angles.append(angle)
    ms.append(p) #add the pth projection
    #~ original_angles.append( complex(0.0, 1.0) )
    original_angles.append( complex(1.0, 0.0) )
        
    for angle in original_angles:
        rotated_angle = angle*rotation
        #~ rotated_angle = complex(angle.imag*rotation.imag - angle.real*rotation.real, angle.imag*rotation.real + rotation.imag*angle.real) #Imants' coordinate system
        #~ rotated_angle = complex(angle.imag*rotation.real + rotation.imag*angle.real, angle.imag*rotation.imag - angle.real*rotation.real)
        #~ rotated_angle = rotation*angle
        '''if abs(rotated_angle.real) >= p or abs(rotated_angle.imag) >= p: # only for coomparison, since negative values would be lost otherwise
            rotated_angle_rational = complex( int(rotated_angle.real)%p, int(rotated_angle.imag)%p )
        else:
            rotated_angle_rational = rotated_angle'''
        rotated_angle_rational = complex( int(rotated_angle.real+p)%p, int(rotated_angle.imag+p)%p )
        rotated_angles.append(rotated_angle_rational)
        
        #compute m'
        #~ u = nt.minverse(rotated_angle_rational.real, p)
        u = nt.minverse(rotated_angle_rational.imag, p)
        #~ inverse = (u*int(rotated_angle_rational.real))%p
        inverse = (u*int(rotated_angle_rational.imag))%p
        inverses.append(inverse)
        if inverse > 1:
            print("MInverse Failed")
        elif inverse == 0:
            m_dash = p
        else:
            #~ m_dash = (  u * int(rotated_angle_rational.imag) )%p
            m_dash = (  u * int(rotated_angle_rational.real) )%p
        m_dashs.append(m_dash)
        
    #~ print original_angles
    #~ print rotated_angles
    #~ print inverses
    #~ print ms
    #~ print m_dashs
        
    return m_dashs, rotated_angles

def fareyMapping(N):
    '''
    Print out the Farey/rational angle mapping to the finite angle set.
    The angles are based on L1 norm minimal rational angle set.
    Also returns the finite angle set, matching rational angles and No. of bins lists
    '''
    #create angle set with Farey vectors
    fareyVectors = farey.Farey()        
    fareyVectors.compactOn()
    fareyVectors.generateFiniteWithCoverage(N)
#    angles = fareyVectors.vectors
#    print "Number of Projections:", len(angles)
#    print fareyVectors.finiteAngles
#    print fareyVectors.vectors
    
    #sort to reorder result for prettier printing
    finiteAnglesSorted, anglesSorted = fareyVectors.sort('finite')
    
    #print mapping
    BList = []
    for finiteAngle, angle in zip(finiteAnglesSorted, anglesSorted):
        p, q = farey.get_pq(angle)
        B = farey.projectionLength(angle, N, N)
        BList.append(B)
        print("m:", finiteAngle, "p:", p, "q:", q, "B:", B)
        
    return finiteAnglesSorted, anglesSorted, BList

def noise(bins, SNR=0.95):
    '''
    Return (Gaussian) noise of DRT bins as Normal(bins[j],SNR*bins[j]).
    You can then multiply or add this to the bins. Noise is not quantised, do it yourself with astype(np.int32)
    '''
    noise = np.zeros(bins.shape)
    for m, row in enumerate(bins):
        for t, bin in enumerate(row):
            noise[m,t] = random.normalvariate(bin, 0.15*(1.0-SNR)*bin)-bin
            
    return noise
    
#-------------
#filters
def oversampling(n, p = 2):
    '''
    Produce the nxn oversampling filter that is needed to exactly filter dyadic DRT etc.
    '''
    gcd_table = np.zeros(n)

    gcd_table[0] = n + n/p
    gcd_table[1] = 1
    for j in range(2,n):
        u, v, gcd_table[j] = nt.extended_gcd( int(j), int(n) )

    filter = np.zeros((n,n))
    for j in range(0,n):
        for k in range(0,n):
            if gcd_table[j] < gcd_table[k]:
                filter[j,k] = gcd_table[j]
            else:
                filter[j,k] = gcd_table[k]

    return filter
    
def oversamplingFilter(n, M, p = 2):
    '''
    Produce the nxn oversampling filter that is needed to exactly filter dyadic DRT etc.
    This version returns values as multiplicative inverses (mod M) for use with the NTT
    '''
    gcd_table = np.zeros(n)
    gcdInv_table = np.zeros(n)

    gcd_table[0] = n + int(n)/p
    gcdInv_table[0] = nt.minverse(gcd_table[0], M)
    gcd_table[1] = 1
    gcdInv_table[1] = nt.minverse(gcd_table[1], M)
    for j in range(2,n):
        u, v, gcd_table[j] = nt.extended_gcd( int(j), int(n) )
        gcdInv_table[j] = nt.minverse(gcd_table[j], M)

    filter = nt.zeros((n,n))
    for j in range(0,n):
        for k in range(0,n):
            if gcd_table[j] < gcd_table[k]:
                filter[j,k] = nt.integer(gcdInv_table[j])
            else:
                filter[j,k] = nt.integer(gcdInv_table[k])

    return filter

def oversampling_1D(n, p = 2, norm = False):
    '''
    The 1D oversampling. All values are GCDs.
    Use the filter versions to remove the oversampling.
    '''
    gcd_table = np.zeros(n)

    gcd_table[0] = n + int(n)/p
    gcd_table[1] = 1
    for j in range(2,n):
        u, v, gcd_table[j] = nt.extended_gcd( int(j), int(n) )

    return gcd_table

def oversampling_1D_filter(n, p = 2, norm = False):
    '''
    The 1D filter for removing oversampling. All values are multiplicative inverses.
    To apply this filter multiply with 1D FFT slice
    '''
    normValue = 1
    if norm:
        normValue = n
        
    gcd_table = np.zeros(n)
    gcdInv_table = np.zeros(n)

    gcd_table[0] = n + int(n)/p
    gcdInv_table[0] = 1.0/gcd_table[0]
    gcd_table[1] = 1
    gcdInv_table[1] = 1.0/gcd_table[1]
    for j in range(2,n):
        u, v, gcd_table[j] = nt.extended_gcd( int(j), int(n) )
        gcdInv_table[j] = 1.0/(gcd_table[j]*normValue)

    return gcdInv_table

def oversampling_1D_filter_Integer(n, M, p = 2, norm = False):
    '''
    The 1D filter for removing oversampling. All values are multiplicative inverses.
    To apply this filter multiply with 1D NTT slice
    '''
    gcd_table = np.zeros(n)
    gcdInv_table = nt.zeros(n)
    inv = nt.integer(1)
    if norm:
        inv = nt.integer(nt.minverse(int(n), M))

    gcd_table[0] = n + int(n)/p
    gcdInv_table[0] = nt.integer(nt.minverse(gcd_table[0], M))
    gcd_table[1] = 1
    gcdInv_table[1] = nt.integer(nt.minverse(gcd_table[1], M))
    for j in range(2,n):
        u, v, gcd_table[j] = nt.extended_gcd( int(j), int(n) )
        gcdInv_table[j] = (nt.integer(nt.minverse(gcd_table[j], M))*inv)%M

    return gcdInv_table
    
def convolve(x, y):
    '''
    Compute the nD convolution of two real arrays of the same size.
    '''
    xHat = fftpack.fftn(x+0j)
    yHat = fftpack.fftn(y+0j)
    xHat *= yHat
    
    return np.real(fftpack.ifftn(xHat))

def deconvolve(star, psf):
    '''
    Generic deconvolution function
    '''
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    
    return np.real( fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft/psf_fft))) )
    