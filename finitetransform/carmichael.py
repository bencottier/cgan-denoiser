# -*- coding: utf-8 -*-
"""
Carmichael number transforms module
"""
import numpy as np
import finitetransform.numbertheory as nt #local modules

MODULUS = np.uint64(np.iinfo(np.uint32).max + 1) #2^M
MAXLENGTH = np.uint64(MODULUS/np.uint64(4)) #2^(M-2)
PRIMITIVEROOT = np.uint64(5)
NTT_FORWARD = -1
NTT_INVERSE = 1

def randint(minValue, maxValue, shape):
    '''
    Convenience member for defining 64-bit integer arrays with random values upto and excluding
    maximum value provided
    '''
    data = nt.zeros(shape)
    for index, value in enumerate(data):
        data[index] = nt.integer(np.random.randint(minValue, high=maxValue, size=None))
    return data
    
def multiply_1D(fctSignal1, fctSignal2, modulus = MODULUS):
    '''
    Multiply two 1D signals (mod M) and return the result as an array
    '''
    result = nt.zeros(fctSignal1.shape)
    for index, value in enumerate(np.nditer(fctSignal1)):
#        print "index:",index
        val1 = nt.integer(fctSignal1[index])
        val2 = nt.integer(fctSignal2[index])
        result[index] = (val1*val2)%modulus
        
    return result
    
def multiply_2D(array1, array2, modulus = MODULUS):
    '''
    Convenience function to multiply two arrays (mod M)
    '''
    result = nt.zeros(array1.shape)
    for x, row in enumerate(array1):
        for y, col in enumerate(row):
            val1 = nt.integer(array1[x,y])
            val2 = nt.integer(array2[x,y])
            result[x,y] = (val1*val2)%modulus
    
    return result

def bigshr(d, s, n):
    '''
    Shifts s right one bit to d, returns carry bit.
    Adapted from Mikko Tommila et al., APFloat Library 2005.
    '''
    b = nt.integer(0)
    t = nt.integer(0)
    tmp = nt.integer(0)
    
    if not n:
        return nt.integer(0)

    d += n
    s += n

    for t in range(0, n):
        d -= nt.integer(1)
        s -= nt.integer(1)

        tmp = (s >> nt.integer(1)) + (0x80000000 if b else nt.integer(0))
        b = s & nt.integer(1)
        d = tmp                              # Works also if d = s

    return b, d

def pow_mod_direct(base, exp, modulus = MODULUS):
    '''
    Compute the power base^exp (mod modulus).
    This computation is as simple and direct as possible.
    '''
    r = nt.integer(1) 
    
    if exp == nt.integer(0):
        return r
        
    for e in nt.arange(0, exp):
        r *= base
        r %= modulus
    
    return r

def pow_mod(base, exp, modulus = MODULUS):
    '''
    Compute the power base^exp (mod modulus).
    Adapted from Mikko Tommila et al., APFloat Library 2005.
    '''
    r = nt.integer(0)    
    b = nt.integer(0) 
#    print 'MODULUS:', modulus, 'base:', base, 'exp:', exp
    
    if exp == nt.integer(0):
        return nt.integer(1)

    b, exp = bigshr(exp, exp, nt.integer(1))
    while not b:
        base = (base * base)%modulus
        b, exp = bigshr(exp, exp, nt.integer(1))

    r = base

    while exp > nt.integer(0):
        base = (base * base)%modulus
        b, exp = bigshr(exp, exp, nt.integer(1))
        if b:
            r = (r * base)%modulus

    return r
    
def ct(data, n, isign = NTT_FORWARD, pr = PRIMITIVEROOT, modulus = MODULUS, maxLength = MAXLENGTH):
    '''
    Compute the direct Carmichael Transform (i.e. not the Fast version).
    Returns the transformed signal.
    '''
    w = nt.integer(0)    
    
    if isign > 0:
        w = pow_mod_direct(pr, nt.integer(maxLength - maxLength / n), modulus) #n must be power of two
    else:
        w = pow_mod_direct(pr, nt.integer(maxLength / n), modulus)    
    print("Base:", w)
    
    result = nt.zeros(n)
    sums = nt.zeros(n)
    for k in nt.arange(0, n):
#        sums[k] = 0
        sums[k] = modulus-n
        for t in nt.arange(0, n):
#            harmonic = pow_mod_direct(w, nt.integer( k*t ), modulus)
            harmonic = pow_mod_direct(w, nt.integer( (2*k+1)*(2*t+1) ), modulus)
            result[k] += ( data[t] * harmonic )%modulus
            result[k] %= modulus
            sums[k] += harmonic
            sums[k] %= modulus
#            print "t:", t, "exp:", nt.integer(k*t), "\tdata[t]:", data[t], "*", harmonic, "=", ( data[t] * harmonic )%modulus, "\tresult[k]:", result[k], "\tsums[k]:", sums[k]
            print("t:", t, "exp:", nt.integer((2*k+1)*(2*t+1)), "\tdata[t]:", data[t], "*", harmonic, "=", ( data[t] * harmonic )%modulus, "\tresult[k]:", result[k], "\tsums[k]:", sums[k])
        print("\n")
    print("Sums:", sums)
    print("Result:", result)
    
#    for k in arange(1, n):
#        sums[k] = modulus-sums[k]
#        if isign > 0:
#            result[k] = (result[k]-sums[k]+modulus)%modulus
#        else:
#            result[k] = (result[k]+sums[k])%modulus
#        
#    print "Sums Corrected:", sums
    
    return result
    
def rearrange(data, n):
    '''
    Bit-reversal of Data of size n inplace. n should be dyadic.
    Adapted from Mikko Tommila et al., APFloat Library 2005.
    '''
    target = nt.integer(0)
    mask = nt.integer(n)
    
    #For all of input signal
    for position in range(0, n):
        #Ignore swapped entries
        if target > position:
            #Swap
            data[position], data[target] = data[target], data[position]

        #Bit mask
        mask = n
        #While bit is set
        mask >>= nt.integer(1)
        while target & mask:
            #Drop bit
            target &= ~mask
            mask >>= nt.integer(1)
        #The current bit is 0 - set it
        target |= mask
    
def fct(data, nn, isign = NTT_FORWARD, pr = PRIMITIVEROOT, modulus = MODULUS, maxLength = MAXLENGTH):
    '''
    Computes the 1D Fast Carmichael Number Theoretic Transform (FCT) using the Cooley-Tukey algorithm.
    The result is NOT normalised within the function.
    
    Default parameters will work fine for dyadic lengths.
    maxLength is normally modulus-1 or modulus/4 depending on type of modulus.
    pr is normally either 3 or 5 depending on modulus.
    
    Other parameters include:
    2113929217, 3 for lengths upto 2^25 as M=63*2^25+1
    2147473409, 3
    
    The transform is done inplace, destroying the input. 
    '''
    w = wr = wt = nt.integer(0)
    wtemp = nt.integer(0)
    istep = i = m = nt.integer(0)

    if isign > 0:
        w = pow_mod(pr, nt.integer(maxLength - maxLength / nn), modulus) #nn must be power of two
    else:
        w = pow_mod(pr, nt.integer(maxLength / nn), modulus)

    rearrange(data, nn)

    mmax = nt.integer(1)
    while nn > mmax:
        istep = mmax << nt.integer(1)
        wr = wt = pow_mod(w, nt.integer(nn / istep), modulus)

        #Optimize first step when wr = 1
        for i in range(0, nn, istep):
            j = int(i + mmax)
            wtemp = data[j]
            data[j] = (data[i] + modulus - wtemp) if data[i] < wtemp else (data[i] - wtemp)
#            data[j] = (data[i] - wtemp)%modulus #causes underflow sometimes
            data[i] = (data[i] + wtemp)%modulus

        for m in range(1, mmax):
            for i in range(m, nn, istep):
                j = int(i + mmax)
                wtemp = (wr * data[j])%modulus #double width for integer multiplication
                data[j] = (data[i] + modulus - wtemp) if data[i] < wtemp else (data[i] - wtemp)
#                data[j] = (data[i] - wtemp)%modulus #causes underflow sometimes
                data[i] = (data[i] + wtemp)%modulus 
            wr = (wr * wt)%modulus #double width for integer multiplication
        mmax = istep
        
    return data
    
def fct_2D(data, nn, isign = NTT_FORWARD, pr = PRIMITIVEROOT, modulus = MODULUS, maxLength = MAXLENGTH):
    '''
    Computes the 2D Carmichael transform for a 2D square array of size nn
    '''
#    d = euclidean((nttw_big_integer)nn,MODULUS,&inv,&y); #Multi Inv of p-1
#    inv = (inv + MODULUS)%MODULUS; #Ensure x is positive

    #Transform Rows
    for row in data: #done in place
        fct(row, nn, isign, pr, modulus, maxLength)

    #Transform Columns
    for column in data.T: #done in place, transpose is cheap
#        for (k = 0; k < nn; k ++)
#            ptrResult[k] = (data[k*nn+j] * inv)%MODULUS; #Stops modulo overrun, div by N early

        fct(column, nn, isign, pr, modulus, maxLength)

#        for (k = 0; k < nn; k ++) #Inverse so Copy and Norm
#            result[k*nn+j] = ptrResult[k];
    return data
    
def ifct(data, nn, pr = PRIMITIVEROOT, modulus = MODULUS, maxLength = MAXLENGTH):
    '''
    Convenience function for inverse 1D Fast Carmichael Transforms. See fct documentation for more details.
    '''
    return fct(data, nn, NTT_INVERSE, pr, modulus, maxLength)

def ifct_2D(data, nn, pr = PRIMITIVEROOT, modulus = MODULUS, maxLength = MAXLENGTH):
    '''
    Convenience function for inverse 2D Fast Carmichael Transforms. See fct_2D documentation for more details.
    '''
    return fct_2D(data, nn, NTT_INVERSE, pr, modulus, maxLength)
    
def harmonics(n, isign = NTT_FORWARD, pr = PRIMITIVEROOT, modulus = MODULUS, maxLength = MAXLENGTH):
    '''
    Computes the harmonics of the 1D Carmichael Number Transform (FCT)
    The returned array then contains the basis functions for each k for the transform.
    '''
    w = nt.integer(0)    
    
    if isign > 0:
        w = pow_mod_direct(pr, nt.integer(maxLength - maxLength / n), modulus) #nn must be power of two
    else:
        w = pow_mod_direct(pr, nt.integer(maxLength / n), modulus)    
    print("Base:", w)
    
#    x = np.random.randint(0, n, n)
#    x = arange(0, n)
#    print "x:", x
    
    result = nt.zeros(n)
    harmonics = nt.zeros( (n, n) )
    for k in nt.arange(0, n):
        for t in nt.arange(0, n):
#            value = nt.integer( x[t]*pow_mod(w, nt.integer(k*t), modulus) )
            value = pow_mod_direct(w, nt.integer(k*t), modulus)
#            value = pow_mod_direct(w, nt.integer((k+1)*(t+1)), modulus)
#            value = pow_mod(w, nt.integer(k*t), modulus)
            value %= modulus
#            if value in harmonics[k]:
#                print "Error: Harmonic not unique"
            harmonics[k, t] = value
            result[k] += value
            result[k] %= modulus
#            print "k, harmonic k: ", k, harmonics[k, t]
#            print "k, t, kt, powmod =", k, t, nt.integer(k*t), pow_mod_direct(w, nt.integer(k*t), modulus)
#        print harmonics[k]
#        print "sum, freq:", harmonics[k].sum(), harmonics[k].sum()%modulus
        
    return result, harmonics
    
def norm_1D(data, N, modulus = MODULUS):
    '''
    Normalise the signal given a full forward and inverse transform (mod M)
    '''
    normData = nt.zeros(N)
    Ninv = nt.integer(nt.minverse(N, modulus)%modulus)
    
    if modulus%2 == 1: #odd number then, modulus likely prime and therefore a field
        for i, value in enumerate(np.nditer(data)):
            normData[i] = (value*Ninv)%modulus
    else: #only ring
        for i, value in enumerate(np.nditer(data)):
            normData[i] = value/N
        
    return normData
    
def norm_2D(data, N, modulus = MODULUS):
    '''
    Normalise the signal given a full forward and inverse transform (mod M).
    Assumes data is a square array of size N
    '''
    normData = norm_1D(data.flatten(), N*N, modulus)
        
    return normData.reshape((N,N))

def toPixelRange(data, N, modulus = MODULUS):
    '''
    Renormalise the gray scales so that it is easily displayed.
    '''
#    maxValue = int(data.max())
    intData = data.astype(np.int64)
    intData[intData>(modulus/2)] -= modulus
    
    return intData