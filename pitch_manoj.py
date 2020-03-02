# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 10:01:21 2019

@author: Hirak Das Gupta
"""

import numpy as np
import scipy as sp


def pitch_of_sound(data, fs):
    drc, spdl = A_LawDRC(data, fs)
    he, ht, sd = Hlbrt_fnc(drc, fs)
    pk = Pk_dtct(he, fs)
    pk1 = pk.reshape(np.size(pk), 1)
    AR1, AR2, PkMd_mm1, AR2 = median_meancombo1(pk1, 11, 5, 11, 5)
    My_diff = Dff_My(np.squeeze(PkMd_mm1))
    My_diff = np.squeeze(My_diff)
    tmwndw = int(15e-3 * fs)
    gci_reg, imptr = Kad_exp_calc(My_diff, tmwndw, fs)
    ptchFrq = fs / np.diff(gci_reg)
    ptch_Frq_f = np.zeros(int(np.size(ptchFrq) + 1))
    ptch_Frq_f[0: int(np.size(ptch_Frq_f) - 1)] = ptchFrq
    ptch_Frq_f[int(np.size(ptch_Frq_f)) - 1] = ptch_Frq_f[int(np.size(ptch_Frq_f)) - 2]
    tmstmp = gci_reg / fs
    Tmspch = np.linspace(0, int(np.size(spdl) - 1), num=int(np.size(spdl)))
    Tmspch = Tmspch / fs

    return tmstmp, ptch_Frq_f


## Hilbert envelope 
def Hlbrt_fnc(Mysig, fs):
    FIRwindwlen = 15e-3
    totallen = round(fs * FIRwindwlen)
    if (totallen % 2 == 0):
        totallen = totallen + 1
    Filterlength = totallen
    Nx = (Filterlength - 1) / 2

    w = np.hamming(Filterlength)
    n = np.arange(start=0, stop=Filterlength, step=1)
    n1 = n - Nx
    impulseresp = np.zeros(Filterlength)
    for My_indx in n1:
        if (My_indx != 0):
            impulseresp[int(My_indx + Nx)] = (2 / (np.pi * My_indx)) * ((np.sin(My_indx * np.pi / 2)) ** 2)
        else:
            impulseresp[int(My_indx + Nx)] = 0

    impulse_hamming = impulseresp * w
    Mysig = np.squeeze(Mysig)
    filter_op = sp.signal.lfilter(impulse_hamming, [1.0], Mysig)
    filter_op1 = filter_op[int(Nx): np.size(filter_op)]
    Mysig1 = Mysig[0:np.size(filter_op1)]
    My_Hilbenv = (filter_op1 ** 2) + (Mysig1 ** 2)
    return (My_Hilbenv, filter_op1, Mysig1)


## Dynamic range compression usinbg A law
def A_LawDRC(spch, fs):
    x = spch
    y = x * x
    tmwndw2 = round(25e-3 * fs)
    h2 = np.hamming(tmwndw2)
    h2 = h2 ** 2
    h2 = h2 / sum(h2)
    fltop2 = sp.signal.lfilter(h2, [1.0], np.squeeze(y))
    fltop2 = fltop2[round(tmwndw2 / 2): np.size(fltop2)]
    fltop2 = np.sqrt(fltop2)
    y = y[0: np.size(fltop2)]
    x = x[0: np.size(fltop2)]
    fltop2 = fltop2 * 2.5  # To make the resulting output to be in the range of 1.
    A = 40
    ##    i=np.linspace(start=0, stop= np.size(fltop2)-1, num = np.size(fltop2))
    alpha = np.zeros((np.size(fltop2), 1))
    gain_calc = np.zeros((np.size(fltop2), 1))
    DRC_speech = np.zeros((np.size(fltop2), 1))
    i = 0
    for Ldat in fltop2:
        if (Ldat <= 1 / A):
            alpha[i] = (A * Ldat) / (1 + np.log(A))
            gain_calc[i] = A / (1 + np.log(A))
        else:
            alpha[i] = (1 + np.log(A * Ldat)) / (1 + np.log(A))
            gain_calc[i] = alpha[i] / Ldat
        DRC_speech[i] = gain_calc[i] * x[i]
        i = i + 1
    return (DRC_speech, x)


## Peak and valley detection
def Pk_dtct(Henv, fs):
    Henv = np.squeeze(Henv)
    alpha3 = 0.1
    beta_13 = 0.0059 ** (1 / (10e-3 * fs))
    # beta_13=0.95; #for 10 k Hz Sampling Frequency
    # beta_13=0.9684; # For 16 KHz Sampling Freq.
    vold3 = 0
    pold3 = 0

    p3 = np.zeros((np.size(Henv), 1))
    v3 = np.zeros((np.size(Henv), 1))

    p3 = np.squeeze(p3)
    v3 = np.squeeze(v3)

    i = 0
    for pcar in Henv:
        # Peak variable updation
        if (pcar >= pold3):
            p3[i] = (alpha3 * pold3) + (1 - alpha3) * pcar
        else:
            p3[i] = (beta_13 * pold3) + (1 - beta_13) * vold3
        # Valley variable updation
        if (pcar <= vold3):
            v3[i] = (alpha3 * vold3) + (1 - alpha3) * pcar
        else:
            v3[i] = (beta_13 * vold3) + (1 - beta_13) * pold3
        pold3 = p3[i]
        vold3 = v3[i]
        i += 1
    return (p3)


## Median filter of fixed length
def Med_flt(inpdat, fltlen):
    x1 = np.zeros((np.size(inpdat) + fltlen - 1, 1))
    x1[fltlen - 1: np.size(x1)] = inpdat[0:np.size(inpdat)]
    y = np.zeros((np.size(inpdat)))
    cnt = 0
    stp = 0
    endpt = fltlen

    while endpt <= np.size(x1):
        ll = np.asarray(x1[stp:endpt])
        #        ll=np.squeeze(x1[stp:endpt])
        y[cnt] = np.median(ll)
        stp += 1
        endpt += 1
        cnt += 1

    MedFilt = y
    MedFilt = MedFilt[round((fltlen - 1) / 2): np.size(MedFilt)]  # Assuming the filter length to be odd
    SPD_med = inpdat[0: np.size(MedFilt)]
    return (MedFilt, SPD_med)


## Mean Filter     
def Mean_flt(ipnd, mnlen):
    Filt_length = mnlen
    Nx = (Filt_length - 1) / 2  # Assuming odd Filter length
    impulseresp = np.ones(mnlen)
    impulseresp = impulseresp / Filt_length
    Mnfltop = sp.signal.lfilter(impulseresp, [1.0], np.squeeze(ipnd))
    Mnfltop = Mnfltop[int(Nx):np.size(Mnfltop)]
    opdl = ipnd[0: np.size(Mnfltop)]
    return (Mnfltop, opdl)


## Median mean filter
def median_meancombo1(inputdata, MedianLen1, MeanLen1, MedianLen2, MeanLen2):
    inp_spech = inputdata
    Medfiltop, DR1 = Med_flt(inp_spech, MedianLen1)
    mean_medianop, DR2 = Mean_flt(Medfiltop, MeanLen1)
    mean_medianop = mean_medianop.reshape(np.size(mean_medianop), 1)
    dalay1 = ((MedianLen1 - 1) / 2) + ((MeanLen1 - 1) / 2)
    Delayin_sp = inputdata[0: np.size(inputdata) - int(dalay1)]
    FirstlevelDelay = Delayin_sp
    Myinterim = mean_medianop
    z = (Delayin_sp) - (mean_medianop)
    # Second block of median mean
    v1, DR3 = Med_flt(z, MedianLen2)
    v, DR4 = Mean_flt(v1, MeanLen2)
    v = v.reshape(np.size(v), 1)
    dalay2 = ((MedianLen2 - 1) / 2) + ((MeanLen2 - 1) / 2)
    Delayin_y = Myinterim[0: np.size(Myinterim) - int(dalay2)]
    w = Delayin_y + v
    Finalop_ckt = w
    Delay_inp = inputdata[0: np.size(inputdata) - int(dalay1) - int(dalay2)]
    return (mean_medianop, FirstlevelDelay, Finalop_ckt, Delay_inp)


## Function for 5 point differentiation
def Dff_My(MyMM_op):
    ah11 = np.array([-1 / 12, 8 / 12, 0, -8 / 12, 1 / 12])
    Sle_op = sp.signal.lfilter(ah11, [1.0], np.squeeze(MyMM_op))
    aa = np.convolve(ah11, np.squeeze(MyMM_op))
    Sle_op = Sle_op[2:np.size(Sle_op)]
    Sle_op = Sle_op.reshape(np.size(Sle_op), 1)
    return (Sle_op)


# finding a maximum and its index    
def mymaxfunc(my_ind):
    mymax = my_ind[0]
    max_ind = 0
    k = 0
    for myval in my_ind:
        if (myval > mymax):
            mymax = myval
            max_ind = k
        k = k + 1
    return (mymax, max_ind)


# finding peak of maximu sum subarray by kadane
def Kad_exp_calc(My_diff, tmwndw, fs):
    My_diff_prc = np.squeeze(My_diff)
    My_diff_prc = np.zeros(int(np.size(My_diff) + 1 + tmwndw))
    My_diff_prc[0: np.size(My_diff)] = My_diff
    fnlendpt = int(tmwndw)
    fnlstpt = 0
    nxx = np.linspace(0, int(tmwndw - 1), num=int(tmwndw))
    #    Tau=100 # For 10 kHz
    Tau = 160;  # For 16 kHz
    mywndw = np.exp(-nxx / Tau)
    imptr = np.zeros(np.size(My_diff_prc))

    gci_reg = -100 * np.ones(np.size(My_diff_prc))
    imptr_pt = 0
    while fnlendpt <= np.size(My_diff_prc):
        tempdata = My_diff_prc[int(fnlstpt): int(fnlendpt)]
        tmp11 = tempdata
        tempdata = tempdata * mywndw

        # kadane algorithm

        max_curr = 0
        max_global = -5

        stpt = 0
        endpt = 0
        dumpt = 0  # in matlab it is 1 due to indexing difference

        lpindx = 0  # in matlab it is 1 due to indexing difference

        for mydum in tempdata:
            max_curr = max_curr + mydum
            if (max_curr > max_global):
                max_global = max_curr
                stpt = dumpt
                endpt = lpindx

            if (max_curr < 0):
                max_curr = 0
                dumpt = lpindx + 1

            lpindx = lpindx + 1
        # Kadane ends here
        # np.size(My_diff_prc):
        if (stpt == endpt):
            endpt = endpt + 1  # This line is not there in matlab . This is due to indexing issue in python
        [mxval, ptintrst] = mymaxfunc(tmp11[stpt: endpt])

        gci_reg[imptr_pt] = int(fnlstpt + stpt + ptintrst)
        imptr_pt = imptr_pt + 1

        imptr[fnlstpt + stpt + ptintrst] = 1
        fnlstpt = int(fnlstpt + stpt + ptintrst + int(1.5e-3 * fs))  # 2ms as refractory period
        fnlendpt = fnlstpt + int(tmwndw)  # Have to see to put -1 or not

        if (imptr_pt >= 5):
            ptchsamp = np.diff(gci_reg[imptr_pt - 5: imptr_pt - 1])
            if (np.mean(ptchsamp) <= (15e-3 * fs) and np.mean(ptchsamp) >= (2e-3 * fs)):
                TauN = round(0.666 * np.mean(ptchsamp))
                Tau = (0.9 * TauN) + (0.1 * Tau)
                mywndw = np.exp(-nxx / Tau)
        else:
            mywndw = np.exp(-nxx / Tau)

    gci_regF = gci_reg[0:imptr_pt]
    imptrF = imptr[0: np.size(My_diff)]
    return (gci_regF, imptrF)
