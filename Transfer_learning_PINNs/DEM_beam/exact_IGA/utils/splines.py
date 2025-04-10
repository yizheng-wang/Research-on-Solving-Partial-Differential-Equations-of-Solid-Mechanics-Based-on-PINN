#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utitlities for manipulating splines that are not in Geomdl package
"""
import numpy as np
from scipy.special import binom


def bspdegelev(d, c, k, t):
    """
    Degree elevate a univariate B-Spline. 
    (see  https://octave.sourceforge.io/nurbs/function/bspdegelev.html)

    Parameters
    ----------
    d : Degree of the B-Spline.
    c : Control points, matrix of size (dim,nc).
    k : Knot sequence, row vector of size nk.
    t : Raise the B-Spline degree t times.

    Returns
    -------
    ic : Control points of the new B-Spline. 
    ik : Knot vector of the new B-Spline.

    """
    mc, nc = c.shape
    ic = np.zeros((mc, nc * (t + 1)))
    n = nc - 1
    bezalfs = np.zeros((d + 1, d + t + 1))
    bpts = np.zeros((mc, d + 1))
    ebpts = np.zeros((mc, d + t + 1))
    Nextbpts = np.zeros((mc, d + 1))
    alfs = np.zeros((d, 1))

    m = n + d + 1
    ph = d + t
    ph2 = np.floor(ph / 2).astype(int)

    # Compute Bezier degree elevation coefficients
    bezalfs[0, 0] = 1.0
    bezalfs[d, ph] = 1.0

    for i in range(1, ph2 + 1):
        inv = 1 / binom(ph, i)
        mpi = min(d, i)

        for j in range(max(0, i - t), mpi + 1):
            bezalfs[j, i] = inv * binom(d, j) * binom(t, i - j)

    for i in range(ph2 + 1, ph):
        mpi = min(d, i)
        for j in range(max(0, i - t), mpi + 1):
            bezalfs[j, i] = bezalfs[d - j, ph - i]

    mh = ph
    kind = ph + 1
    r = -1
    a = d
    b = d + 1
    cind = 1
    ua = k[0]

    ic[0:mc, 0] = c[0:mc, 0]
    ik = (ua * np.ones(ph + 1)).tolist()

    # Initalize the first Bezier segment
    bpts = c.copy()

    # Big loop through knot vector
    while b < m:
        i = b
        while b < m and k[b] == k[b + 1]:
            b += 1
        mul = b - i + 1
        mh = mh + mul + t
        ub = k[b]
        oldr = r
        r = d - mul

        # Insert knot u[b] r times
        if oldr > 0:
            lbz = np.floor((oldr + 2) / 2).astype(np.int)
        else:
            lbz = 1

        if r > 0:
            rbz = np.floor((r + 1) / 2).astype(np.int)
        else:
            rbz = ph

        if r > 0:
            # Insert knot to get Bezier segment
            numer = ub - ua
            for q in range(d, mul, -1):
                alfs[q - mul - 1] = numer / (k[a + q] - ua)

            for j in range(1, r + 1):
                save = r - j
                s = mul + j

                for q in range(d, s - 1, -1):
                    for ii in range(0, mc):
                        tmp1 = alfs[q - s] * bpts[ii, q]
                        tmp2 = (1 - alfs[q - s + 1]) * bpts[ii, q - 1]
                        bpts[ii, q] = tmp1 + tmp2

                Nextbpts[:, save] = bpts[:, d]
        # End of insert knot

        # Degree elevate Bezier
        for i in range(lbz, ph + 1):
            ebpts[:, i] = np.zeros(mc)
            mpi = min(d, i)
            for j in range(max(0, i - t), mpi + 1):
                for ii in range(0, mc):
                    tmp1 = ebpts[ii, i]
                    tmp2 = bezalfs[j, i] * bpts[ii, j]
                    ebpts[ii, i] = tmp1 + tmp2
        # End of degree elevating Bezier

        if oldr > 1:
            # Must remove knot u=k[a] oldr times
            first = kind - 2
            last = kind
            den = ub - ua
            bet = np.floor((ub - ik[kind - 1] / den)).astype(np.int)

            # Knot removal loop
            for tr in range(1, oldr):
                i = first
                j = last
                kj = j - kind + 1
                while j - i > tr:
                    # Loop and compute the new control points
                    if i < cind:
                        alf = (ub - ik[i]) / (ua - ik[i])
                        tmp1 = alf * ic[:, i]
                        tmp2 = (1 - alf) * ic[:, i - 1]
                        ic[:, i] = tmp1 + tmp2
                    if j >= lbz:
                        if j - tr <= kind - ph + oldr:
                            gam = (ub - ik[j - tr]) / den
                            tmp1 = gam * ebpts[:, kj]
                            tmp2 = (1 - gam) * ebpts[:, kj + 1]
                            ebpts[:, kj] = tmp1 + tmp2
                        else:
                            tmp1 = bet * ebpts[:, kj]
                            tmp2 = (1 - bet) * ebpts[:, kj + 1]
                            ebpts[:, kj] = tmp1 + tmp2
                    i += 1
                    j -= 1
                    kj -= 1
                first -= 1
                last += 1
        # End of removing knot n=k[a]

        # Load the knot ua
        if a != d:
            for i in range(0, ph - oldr):
                ik.append(ua)
                kind += 1

        for j in range(lbz, rbz + 1):
            for ii in range(0, mc):
                ic[ii, cind] = ebpts[ii, j]
            cind += 1

        if b < m:
            # Setup for next pass through loop
            bpts[:, 0:r] = Nextbpts[:, 0:r]
            bpts[:, r : d + 1] = c[:, b - d + r : b + 1]
            a = b
            b += 1
            ua = ub
        else:
            for i in range(0, ph + 1):
                ik.append(ub)
    # End big while loop
    ic = ic[:, 0:cind]
    return ic, ik


def findspan(n, u, knot):
    """
     Find the span of a B-Spline knot vector at a parametric point

    Parameters
    ----------
    n : (int) number of control points - 1
    u : (list of reals) the parameteric values
    knot : (list of reals) the knot vector

    Returns
    -------
    s : (list of ints) the knot span index or indices
    
    """
    if np.min(u) < knot[0] or np.max(u) > knot[-1]:
        raise ValueError("Some value is outside the knot span")
    knot_arr = np.array(knot)
    s = np.zeros(len(u), dtype=int)
    for j in range(len(s)):
        if u[j] == knot[n + 1]:
            s[j] = n
            continue
        s[j] = np.argwhere(knot_arr <= u[j])[-1]
    return s


def bspkntins(d, c, k, u):
    """
    Insert knots into a B-Spline 

    Parameters
    ----------
    d : (int) spline degree
    c : (array mc x nc) control points
    k : (list nk) knot sequence
    u : (list nu) new knots        

    Returns
    -------
    ic : (array mc, nc + nu) new control points
    ik : (list nk+nu)  new knot sequence

    """
    tol_eq = 1e-10
    mc, nc = c.shape
    u = np.sort(u)
    k = np.array(k)
    nu = len(u)
    nk = len(k)
    
    ic = np.zeros((mc, nc+nu))
    ik = np.zeros(nk + nu)
    
    n = nc - 1
    r = nu - 1
    
    m = n + d + 1
    a = findspan(n, [u[0]], k)[0]
    b = findspan(n, [u[r]], k)[0]
    b += 1
    
    ic[:, 0:a-d+1] = c[:, 0:a-d+1]
    ic[:, b+nu-1:nc+nu] = c[:, b-1:nc]
    
    ik[0:a+1] = k[0:a+1]
    ik[b+d+nu:m+nu+1] = k[b+d:m+1]
    
    ii = b + d - 1
    ss = ii + nu
    
    for jj in range(r, -1, -1):
        ind = np.arange(a+1,ii+1)
        temp = np.where(k[ind]>=u[jj])
        ind = ind[temp]
        ic[:, ind+ss-ii-d-1] = c[:, ind-d-1]
        ik[ind+ss-ii] = k[ind]
        ii -= len(ind)
        ss -= len(ind)
        
        ic[:,ss-d-1] = ic[:, ss-d]
        for l in range(1, d+1):
            ind = ss - d + l
            alfa = ik[ss+l] - u[jj]
            if abs(alfa)<tol_eq:
                
                ic[:, ind-1] = ic[:, ind]
            else:
                alfa = alfa/(ik[ss+l]-k[ii-d+l])
                tmp = (1-alfa)*ic[:, ind]
                ic[:, ind-1] = alfa*ic[:, ind-1] + tmp
        ik[ss] = u[jj]
        ss -= 1
    return ic, ik
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    