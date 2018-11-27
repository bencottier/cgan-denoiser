# -*- coding: utf-8 -*-
"""
Number Theoretic functions

@author: shakes

Also consists of numbthy module:
Note: Version 0.7 changes some function names to align with SAGE
Author: Robert Campbell, <campbell@math.umbc.edu>
License: Simplified BSD (see details at bottom)

Also consists of a Gaussian Integer class
Based of Robert Campbell, but adapted to use the complex type internally via sub-classing
"""
#import types # Use IntType, LongType for Gaussian Integer class
import cmath
import numpy as np

def integer(value):
    '''
    Declare integer of appropriate type to use with Carmichael transforms.
    Normally this is a 64-bit unsigned integer that numpy supports.
    '''
    return np.uint64(value)
    
def signed_integer(value):
    '''
    Declare signed integer of appropriate type to use with indexing.
    Normally this is a 64-bit signed integer that numpy supports.
    '''
    return np.int64(value)

def negative_one(N, modulus):
    '''
    Declare negative unit integer of appropriate type to use with Carmichael transforms.
    Normally this is a 64-bit unsigned integer that numpy supports.
    '''
    return modulus/N-1
    
def negative_integer(N, value, modulus):
    '''
    Declare negative integer of appropriate type to use with Carmichael transforms.
    Normally this is a 64-bit unsigned integer that numpy supports.
    '''
    return modulus/N-value
    
def zeros(shape):
    '''
    Convenience member for defining 64-bit integer arrays with zeros suitable for use with
    Carmichael functions.
    '''
    return np.zeros(shape, dtype=np.uint64)
    
def ones(shape):
    '''
    Convenience member for defining 64-bit integer arrays with zeros suitable for use with
    Carmichael functions.
    '''
    return np.ones(shape, dtype=np.uint64)

def arange(minValue, maxValue, step=1):
    '''
    Convenience member for defining 64-bit integer arrays with zeros suitable for use with
    Carmichael functions.
    '''
    return np.arange(minValue, maxValue, step, dtype=np.uint64)

def extended_gcd(a, b):
    '''
    Extended Euclidean algorithm
    Take positive integers a, b as input, and return a triple (x, y, d), such that ax + by = d = gcd(a, b).
    ''' 
    x,y, u,v = 0,1, 1,0
    while a != 0:
        q, r = b//a, b%a
        m, n = x-u*q, y-v*q
        b,a, x,y, u,v = a,r, u,v, m,n
    return x, y, b
    
def is_coprime(a, b):
    ''' Return True if numbers are coprime '''
    x, y ,d = extended_gcd(a, b)
    if d == 1:
        return True
    return False
    
def minverse(x, m):
    '''
    Return multiplicative inverse from a tuple (u, v, d); they are the greatest common divisor d
    of two integers x and y and u, v such that d = x * u + y * v.
    When x and y are coprime, then xu is the modular multiplicative inverse of x modulo y.
    '''
    u, v, d = extended_gcd( int(x), int(m) )
    
    return (u+m)%m

def lcm(numbers):
    '''
    Returns the LCM of a list of integers x
    '''
    x = numbers
#    print "Start:", x
    while x.count(x[0]) != len(x): #check if all elements in list equal
        #zip for paired sorting
        pairs = zip(x, numbers)
        pairs.sort()
        #unzip
        x = [point[0] for point in pairs]
        numbers = [point[1] for point in pairs]
        x[0] += numbers[0]
#        print x
        
    return x[0]
    
def solveCRT(remainders, modulii):
    '''
    Given remainders and modulii, solve for x equiv to all remainders mod modulii using Chinese Remainder Theorem (CRT)
    '''
    multiple = 1
    for modulus in modulii:
        multiple *= modulus
#    print "Multiple:", multiple
    
    inverses = []
    compliments = []
    for remainder, modulus in zip(remainders, modulii):
        compliment = multiple/modulus
        compliments.append(compliment)
        inverse = minverse(compliment, modulus)
        inverses.append(inverse)
#    print "Inverses:", inverses
    
    x = 0
    for remainder, inverse, compliment in zip(remainders, inverses, compliments):
        x += remainder*inverse*compliment
    
    return x%multiple
    
def nearestPrime(n):
    '''
    Return the nearest prime number greater than n. This is done using a search via number of primality tests.
    '''
    p = n
    if n % 2 == 0:
        p += 1
    
    count = 0
    maxAllowed = 1000000
    while not isprime(p) and count < maxAllowed:
        p += 2
        count += 1
        
    return p
    
def quickTwoFactors(n):
    '''
    Factor an even number quickly into two factors. Not sure it works for large numbers.
    '''
    n1 = n
    n2 = 1
    while n2 < n/(2*n2):
        n2 *= 2
        n1 = n/n2
#        print "n1:", n1, "n2:", n2
        if n1%2 == 1:
            break
    return n1, n2
    
def partition(n):
    '''
    \author Jerome Kelleher
    Accellerated ascending composition algorithm
    n is the integer to partition
    '''
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2*x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]
        
def partitionWithLength(n,k,l=1):
    '''n is the integer to partition, k is the length of partitions, l is the min partition element size'''
    if k < 1:
        raise StopIteration
    if k == 1:
        if n >= l:
            yield (n,)
        raise StopIteration
    for i in range(l,n+1):
        for result in partitionWithLength(n-i,k-1,i):
            yield (i,)+result
    
############################################################################
# License: Freely available for use, abuse and modification
# (this is the Simplified BSD License, aka FreeBSD license)
# Copyright 2001-2014 Robert Campbell. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in 
#       the documentation and/or other materials provided with the distribution.
############################################################################
import math  # Use sqrt, floor

def gcd(a,b):
	"""gcd(a,b) returns the greatest common divisor of the integers a and b."""
	if a == 0:
		return abs(b)
	return abs(gcd(b % a, a))
	
def xgcd(a,b):
	"""xgcd(a,b) returns a tuple of form (g,x,y), where g is gcd(a,b) and
	x,y satisfy the equation g = ax + by."""
	a1=1; b1=0; a2=0; b2=1; aneg=1; bneg=1
	if(a < 0):
		a = -a; aneg=-1
	if(b < 0):
		b = -b; bneg=-1
	while (1):
		quot = -(a // b)
		a = a % b
		a1 = a1 + quot*a2; b1 = b1 + quot*b2
		if(a == 0):
			return [b, a2*aneg, b2*bneg]
		quot = -(b // a)
		b = b % a;
		a2 = a2 + quot*a1; b2 = b2 + quot*b1
		if(b == 0):
			return (a, a1*aneg, b1*bneg)
			
def power_mod(b,e,n):
	"""power_mod(b,e,n) computes the eth power of b mod n.  
	(Actually, this is not needed, as pow(b,e,n) does the same thing for positive integers.
	This will be useful in future for non-integers or inverses.)"""
	if e<0: # Negative powers can be computed if gcd(b,n)=1
		e = -e
		b = inverse_mod(b,n)
	accum = 1; i = 0; bpow2 = b
	while ((e>>i)>0):
		if((e>>i) & 1):
			accum = (accum*bpow2) % n
		bpow2 = (bpow2*bpow2) % n
		i+=1
	return accum

def inverse_mod(a,n):
	"""inverse_mod(b,n) - Compute 1/b mod n."""
	(g,xa,xb) = xgcd(a,n)
	if(g != 1): raise ValueError("***** Error *****: {0} has no inverse (mod {1}) as their gcd is {2}, not 1.".format(a,n,g))
	return xa % n
	
def is_prime(n):
	"""is_prime(n) - Test whether n is prime using a variety of pseudoprime tests."""
	if n<0: n=-n  # Only deal with positive integers
	if n<2: return False # 0 and 1 are not prime
	if (n in [2,3,5,7,11,13,17,19,23,29]): return True
	return isprimeE(n,2) and isprimeE(n,3) and isprimeE(n,5)
			
def factor(n):
	"""factor(n) - Return a sorted list of the prime factors of n with exponents."""
	# Rewritten to align with SAGE.  Previous semantics available as factors(n).
	if (abs(n) == 1): return "Unable to factor "+str(n) # Can't deal with units
	factspow = []
	currfact = None
	for thefact in factors(n):
		if thefact != currfact:
			if currfact != None:
				factspow += [(currfact,thecount)]
			currfact = thefact
			thecount = 1
		else:
			thecount += 1
	factspow += [(thefact,thecount)]
	return factspow

def prime_divisors(n):
	"""prime_divisors(n) - Returns a sorted list of the prime divisors of n."""
	return list(set(factors(n)))
	
def euler_phi(n):
	"""eulerphi(n) - Computer Euler's Phi function of n - the number of integers
	strictly less than n which are coprime to n.  Otherwise defined as the order
	of the group of integers mod n."""
	# For each prime factor p with multiplicity n, a factor of (p**(n-1))*(p-1)
	return reduce(lambda a,x:a*(x[0]**(x[1]-1))*(x[0]-1),factor(n),1)
	
def carmichael_lambda(n):
	"""carmichaellambda(n) - Compute Carmichael's Lambda function 
	of n - the smallest exponent e such that b**e = 1 for all b coprime to n.
	Otherwise defined as the exponent of the group of integers mod n."""
	# SAGE equivalent is sage.crypto.util.carmichael_lambda(n)
	thefactors = factors(n)
	thefactors.sort()
	thefactors += [0]  # Mark the end of the list of factors
	carlambda = 1 # The Carmichael Lambda function of n
	carlambda_comp = 1 # The Carmichael Lambda function of the component p**e
	oldfact = 1
	for fact in thefactors:
		if fact==oldfact:
			carlambda_comp = (carlambda_comp*fact)
		else:
			if ((oldfact == 2) and (carlambda_comp >= 4)): carlambda_comp /= 2 # Z_(2**e) is not cyclic for e>=3
			if carlambda == 1:
				carlambda = carlambda_comp
			else:
				carlambda = (carlambda * carlambda_comp)/gcd(carlambda,carlambda_comp)
			carlambda_comp = fact-1
			oldfact = fact
	return carlambda
	
def isprimitive(g,n):
	"""isprimitive(g,n) - Test whether g is primitive - generates the group of units mod n."""
	# SAGE equivalent is mod(g,n).is_primitive_root() in IntegerMod class
	if gcd(g,n) != 1: return False  # Not in the group of units
	order = euler_phi(n)
	if carmichaellambda(n) != order: return False # Group of units isn't cyclic
	orderfacts = prime_divisors(order)
	oldfact = 1
	for fact in orderfacts:
		if fact!=oldfact:
			if pow(g,order/fact,n) == 1: return False
			oldfact = fact
	return True
	
def sqrtmod(a,n):
	"""sqrtmod(a,n) - Compute sqrt(a) mod n using various algorithms.
	Currently n must be prime, but will be extended to general n (when I get the time)."""
	# SAGE equivalent is mod(g,n).sqrt() in IntegerMod class
	if(not isprime(n)): raise ValueError("*** Error ***:  Currently can only compute sqrtmod(a,n) for prime n.")
	if(pow(a,(n-1)/2,n)!=1): raise ValueError("*** Error ***:  a is not quadratic residue, so sqrtmod(a,n) has no answer.")
	return TSRsqrtmod(a,n-1,n)
	
def TSRsqrtmod(a,grpord,p):
	"""TSRsqrtmod(a,grpord,p) - Compute sqrt(a) mod n using Tonelli-Shanks-RESSOL algorithm.
	Here integers mod n must form a cyclic group of order grpord."""
	# Rewrite group order as non2*(2^pow2)
	ordpow2=0; non2=grpord
	while(not ((non2&0x01)==1)):
		ordpow2+=1; non2/=2
	# Find 2-primitive g (i.e. non-QR)
	for g in range(2,grpord-1):
		if (pow(g,grpord/2,p)!=1):
			break
	g = pow(g,non2,p)
	# Tweak a by appropriate power of g, so result is (2^ordpow2)-residue
	gpow=0; atweak=a
	for pow2 in range(0,ordpow2+1):
		if(pow(atweak,non2*2**(ordpow2-pow2),p)!=1):
			gpow+=2**(pow2-1)
			atweak = (atweak * pow(g,2**(pow2-1),p)) % p
			# Assert: atweak now is (2**pow2)-residue
	# Now a*(g**powg) is in cyclic group of odd order non2 - can sqrt directly
	d = invmod(2,non2)
	tmp = pow(a*pow(g,gpow,p),d,p)  # sqrt(a*(g**gpow))
	return (tmp*inverse_mod(pow(g,gpow/2,p),p)) % p  # sqrt(a*(g**gpow))/g**(gpow/2)	
	
################ Internally used functions #########################################

def isprimeF(n,b):
	"""isprimeF(n) - Test whether n is prime or a Fermat pseudoprime to base b."""
	return (pow(b,n-1,n) == 1)
	
def isprimeE(n,b):
	"""isprimeE(n) - Test whether n is prime or an Euler pseudoprime to base b."""
	if (not isprimeF(n,b)): return False
	r = n-1
	while (r % 2 == 0): r //= 2
	c = pow(b,r,n)
	if (c == 1): return True
	while (1):
		if (c == 1): return False
		if (c == n-1): return True
		c = pow(c,2,n)

def factorone(n):
	"""factorone(n) - Find a prime factor of n using a variety of methods."""
	if (is_prime(n)): return n
	for fact in [2,3,5,7,11,13,17,19,23,29]:
		if n%fact == 0: return fact
	return factorPR(n)  # Needs work - no guarantee that a prime factor will be returned
			
def factors(n):
	"""factors(n) - Return a sorted list of the prime factors of n. (Prior to ver 0.7 named factor(n))"""
	if n<0: n=-n  # Only deal with positive integers
	if (is_prime(n)):
		return [n]
	fact = factorone(n)
	if (fact == 1): return "Unable to factor "+str(n) # Can't deal with units
	facts = factors(n/fact) + factors(fact)
	facts.sort()
	return facts

def factorPR(n):
	"""factorPR(n) - Find a factor of n using the Pollard Rho method.
	Note: This method will occasionally fail."""
	for slow in [2,3,4,6]:
		numsteps=2*math.floor(math.sqrt(math.sqrt(n))); fast=slow; i=1
		while i<numsteps:
			slow = (slow*slow + 1) % n
			i = i + 1
			fast = (fast*fast + 1) % n
			fast = (fast*fast + 1) % n
			g = gcd(fast-slow,n)
			if (g != 1):
				if (g == n):
					break
				else:
					return g
	return 1
	
################ Functions renamed to align with SAGE #################
def powmod(b,e,n):
	"""powmod(b,e,n) computes the eth power of b mod n. (Renamed power_mod(b,e,n) in ver 0.7)"""
	return power_mod(b,e,n)
	
def isprime(n):
	"""isprime(n) - Test whether n is prime using a variety of pseudoprime tests. (Renamed is_prime in ver 0.7)"""
	return is_prime(n)
	
def invmod(b,n):
	"""invmod(b,n) - Compute 1/b mod n. (Renamed inverse_mod(b,n) in ver 0.7)"""
	return inverse_mod(b,n)

def eulerphi(n):
	"""eulerphi(n) - Compute Euler's Phi function of n - the number of integers strictly less than n which are coprime to n. 
	(Renamed euler_phi(n) in ver 0.7)"""
	return euler_phi(n)
	
def carmichaellambda(n):
	"""carmichaellambda(n) - Compute Carmichael's Lambda function 
	of n - the smallest exponent e such that b**e = 1 for all b coprime to n.
	Otherwise defined as the exponent of the group of integers mod n. (Renamed carmichael_lambda(n) in ver 0.7)"""
	return carmichael_lambda(n)
 
################ #define Gaussian integer class ################    
class GaussianInteger(complex):
    '''
    An extension of the integer in the complex plane, based on the complex type
    
    Based on the code of Robert Campbell. Designed to sub-class complex type rather than re-implement all functions.
    '''
#    def __new__(cls, *args, **kwargs):
#        selfobj = super(GaussianInteger, cls).__new__(cls)
#        return selfobj
    
    def __init__(self, a = 0, b = 0):
        super().__init__()
        
        self.r = int(a)
        self.i = int(b)
        
#    def __repr__(self):  # Overload conversion used for output
#        return "GaussianInteger(" + str(self.r) + "," + str(self.i) + ")"
        
    def __complex__(self):  # Allow conversion to complex type
        return complex(self.r, self.i)
        
#    @classmethod
    def from_complex(self, z):
#        z.__class__ = GaussianInteger
        return GaussianInteger(z.real, z.imag)
        
    def norm(self):
        '''
        Norm of the current Gaussian Integer. Different from complex type modulus.
        It is defined according to Fermat's two square theorem.
        '''
        return self.real*self.real + self.imag*self.imag
    
    def phase(self):
        '''
        Norm of the current Gaussian Integer. Different from complex type modulus.
        It is defined according to Fermat's two square theorem.
        '''
        return cmath.phase(self)
        
    #overload addition operators
    def __add__(self, summand):  # Overload "+"
        if not isinstance(summand, complex) and not isinstance(summand, GaussianInteger):
            summand = GaussianInteger(summand)  # Force to GaussianInteger
        return self.from_complex(complex.__add__(self, summand))
    
    def __radd__(self, summand):  # Overload "+"
        if not isinstance(summand, complex) and not isinstance(summand, GaussianInteger):
            summand = GaussianInteger(summand)  # Force to GaussianInteger
        return self.from_complex(complex.__radd__(self, summand))
    
    def __iadd__(self, summand):  # Overload "+="
        if not isinstance(summand, complex) and not isinstance(summand, GaussianInteger):
            summand = GaussianInteger(summand)  # Force to GaussianInteger
        self = self.from_complex(complex.__add__(self, summand))
        return self
        
    #overload subtract operators
    def __sub__(self, summand):  # Overload "-" 
        if not isinstance(summand, complex) and not isinstance(summand, GaussianInteger):
            summand = GaussianInteger(summand)  # Force to GaussianInteger
        return self.from_complex(complex.__sub__(self, summand))
        
    def __rsub__(self, summand):  # Overload "-" 
        if not isinstance(summand, complex) and not isinstance(summand, GaussianInteger):
            summand = GaussianInteger(summand)  # Force to GaussianInteger
        return self.from_complex(complex.__rsub__(self, summand))
        
    def __isub__(self, summand):  # Overload "-=" 
        if not isinstance(summand, complex) and not isinstance(summand, GaussianInteger):
            summand = GaussianInteger(summand)  # Force to GaussianInteger
        self = self.from_complex(complex.__sub__(self, summand))
        return self
        
    #overload mulitply operators
    def __mul__(self, multiplier):  # Overload "*"
        if not isinstance(multiplier, complex) and not isinstance(multiplier, GaussianInteger):
            multiplier = GaussianInteger(multiplier)  # Force to GaussianInteger
        return self.from_complex(complex.__mul__(self, multiplier))
    
    def __rmul__(self, multiplier):  # Overload "*"
        if not isinstance(multiplier, complex) and not isinstance(multiplier, GaussianInteger):
            multiplier = GaussianInteger(multiplier)  # Force to GaussianInteger
        return self.from_complex(complex.__rmul__(self, multiplier))
    
    def __imul__(self, multiplier): # Overload "*="
        if not isinstance(multiplier, complex) and not isinstance(multiplier, GaussianInteger):
            multiplier = GaussianInteger(multiplier)  # Force to GaussianInteger
        self = self.from_complex(complex.__mul__(self, multiplier))
        return self
        
    #division operators
    def div(self, divisor):
        '''
        Gaussian Integer Division with rounding for integer conversions.
        '''
        if (type(divisor) is int):
            divisor = GaussianInteger(divisor)  # Coerce if dividing GaussianInteger by integer
        q = complex(self.r,self.i)/complex(divisor.real,divisor.imag)
        return GaussianInteger(int(round(q.real)), int(round(q.imag)))
        
    def __div__(self, divisor):  # Overload "/"
        return self.div(divisor)
        
    def __rdiv__(self, divisor):  # Overload "/" 
        return self.div(divisor)
        
    def __idiv__(self, divisor): # Overload "/="
        self = self.div(divisor)
        return self
        
    #modulus operators
    def mod(self, divisor):
        '''
        Compute x (mod N), where x and N are Gaussian integers
        '''
        return self - divisor*self.div(divisor)
        
    def __mod__(self,divisor):  # Overload "%"  
        return self.mod(divisor)
        
    def __imod__(self,divisor): # Overload "%=" 
        self = self.mod(divisor)
        return self
        
    #greatest common divisor algorithms
    def xgcd(self,other):
        '''
        Extended Euclidean Algorithm for Gaussian Integers
        '''
        quot=GaussianInteger(); a1=GaussianInteger(1,0); b1=GaussianInteger(0,0); a2=GaussianInteger(0,0); 
        b2=GaussianInteger(1,0); a = self; b = other;
        if(b.norm() > a.norm()):
            a,b = b,a  # Swap a and b - need to start with a>b
            a1,b1,a2,b2 = a2,b2,a1,b1 # Swap (a1,b1) with (a2,b2)
        while (1):
            quot = a / b
            a %= b
            a1 -= quot*a2; b1 -= quot*b2
            #print a,b,a1,b1,a2,b2
            if (a == GaussianInteger(0,0)):
                return b, a2, b2
            quot = b / a
            b %= a
            a2 -= quot*a1; b2 -= quot*b1
            #print a,b,a1,b1,a2,b2
            if (b == GaussianInteger(0,0)):
                return a, a1, b1
                
    def gcd(self,other):
        '''
        The greatest common divisor of the Gaussian Integer with respect to the argument.
        '''
        return self.xgcd(other)[0]