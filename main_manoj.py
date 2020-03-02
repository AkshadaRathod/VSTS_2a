# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:05:18 2019

@author: Manoj
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 11:34:40 2019

@author: Manoj Kumar Asati
"""
import numpy as np
from scipy import signal, interpolate
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os, subprocess
import pitch_manoj  ### import file pitch.py from the same directory containing this file
import soundfile as sf

# import time


class vsts:

    # Initializer / Instance Attributes
    def __init__(self, file, lpc_ord=12, warpRatio=1, wshift_time_ms=5, win_scale=2):
        self.file = file
        self.lpc_ord = lpc_ord
        self.warpRatio = warpRatio
        self.win_scale = win_scale
        self.filedata, self.Fs= sf.read(file, dtype='float32') #vsts.read_wav_file(file)
        self.filelen = len(self.filedata)
        self.wshift_time_ms = wshift_time_ms
        self.wshift_samples = round(self.Fs * self.wshift_time_ms * 0.001)
        self.N1 = 1 * warpRatio  # convert frame no. to corresponding sample no.
        self.N2 = (len(self.filedata) - 100) * warpRatio
        self.N3 = self.N2  # storing right endpoint

    @staticmethod
    # the readFile method is used to read the file and to extract sampling frequency Fs, filedata
    def read_wav_file(file):
        Fs, data = wavfile.read(file)  # Fs being the sampling frequency and data is an audio data
        # scale to -1.0 -- 1.0
        if data.dtype == 'int16':
            nb_bits = 16  # -> 16-bit wav files
        elif data.dtype == 'int32':
            nb_bits = 32  # -> 32-bit wav files

        max_nb_bit = float(2 ** (nb_bits - 1))
        filedata = data / (max_nb_bit + 1.0)
        filelen = len(filedata)
        return Fs, filedata, filelen

    @staticmethod
    ## cross correlation function definition
    #  XCORR_FFT Calculates the crosscorrelation of the two input vectors
    #  using the fft based method
    def XCORR_FFT(x, y):
        acr = []
        power = 1
        while power < len(x):  # find the next power of 2 for the signal using signal length
            power = power << 1

        # f = np.fft.fft(x,nextPowerOf2(len(x)))  ## fft of signal x
        f = np.fft.fft(x, power)
        f1 = []
        for elm in f:
            f1 = np.append(f1, elm.real ** 2 + elm.imag ** 2)
        c = np.fft.ifft(f1)  ## inverse fft

        for e in np.arange(len(c) - y, len(c), 1):
            acr = np.append(acr, c[e].real)

        for e in np.arange(0, y + 1, 1):
            acr = np.append(acr, c[e].real)

        return acr

    @staticmethod
    ##Levinson calculation
    # Input 'r' variable of selectedFrame
    # Output as 'k_temp' as column vector of type double call in SelectedFrame
    ##
    def levinson(r):
        # print(r)
        n = len(r)

        a_temp = np.zeros(n)

        a = [1.0]
        k = [0.0]
        a_temp[0] = 1.0

        alpha = r[0]
        for i in np.arange(1, n, 1):
            epsilon = r[i]
            for j in np.arange(1, i, 1):
                epsilon += a[j] * r[i - j]

            a = np.append(a, -epsilon / alpha)
            k = np.append(k, -epsilon / alpha)
            alpha = alpha * (1.0 - k[i] * k[i])

            for j in np.arange(1, i, 1):
                a_temp[j] = a[j] + k[i] * a[i - j]
                ## update a[] array into temporary array */
            for j in np.arange(1, i, 1):
                a[j] = a_temp[j]
                ## update a[] array */

        return k[1:]

    @staticmethod
    # Specgram function using fft
    # output return S, f, t
    def specgram(x, n=None, Fs=2, window=None, noverlap=None):
        try:
            if noverlap is None:
                noverlap = np.ceil(len(window) / 2)

            if window is None:
                window == np.hanning(n)

            if n is None:
                n = 256

            if (len(x) <= len(window)):
                raise ValueError("specgram: segment length must be less than the size of X")

            ## compute window offsets
            win_size = len(window)
            # print(win_size)
            if (win_size > n):
                n = win_size
                Warning("specgram fft size adjusted to %d", n)

            step = int(win_size - noverlap)

            ## build matrix of windowed data slices
            offset = np.arange(0, len(x) - win_size, step)
            # S = np.zeros((n, len(offset)))
            S = np.zeros((len(offset), n))

            for i in np.arange(0, len(offset)):
                # S[0:win_size, i] = x[offset[i]:offset[i]+win_size] * window
                S[i, 0:win_size] = x[offset[i]:offset[i] + win_size] * window

            ## compute Fourier transform
            # S = np.transpose(S)

            S = np.fft.fft(S)

            S = np.transpose(S)

            # S1 = np.fft.fft(S)

            ## extract the positive frequency components
            if np.remainder(n, 2) == 1:
                ret_n = int((n + 1) / 2)
            else:
                ret_n = int(n / 2)

            S = S[0:ret_n, :]
            # print(Fs)
            f = np.arange(0, ret_n) * Fs / n
            t = offset / Fs

            return S, f, t

        except ValueError as e:
            exit(str(e))

    @staticmethod
    ## fit_LSBSpline method
    def fit_LSBSpline(xval, yval, knots, nSplinePts):
        # xval = disp_ag_x
        # yval = a_t
        t = np.cumsum(np.sqrt((np.diff(xval)) ** 2 + (np.diff(yval)) ** 2))
        t = np.array([t])
        if (t.shape[0] == 1):
            t = np.append(0, t)
        else:
            t = np.append(0, np.transpose(t))

        tmax = max(t)
        tmin = min(t)

        tTemp = np.linspace(tmin, tmax, num=nSplinePts)

        tck = interpolate.splrep(x=t, y=xval, k=4)
        Tempx = interpolate.splev(tTemp, tck)  # Evaluate spline function

        tck = interpolate.splrep(x=t, y=yval, k=4)
        Tempy = interpolate.splev(tTemp, tck)

        return Tempx, Tempy

    @staticmethod
    def solafs(X, scale_factor, W=200, Wov=100, Kmax=400, Wsim=100, xdecim=8, kdecim=2):
        #   Y = solafs(X, scale_factor, W, Wov, Kmax, Wsim, xdec, kdec)   Do SOLAFS timescale mod'n
        #   Y is X scaled to run scale_factor x faster.  X is added-in in windows
        #   W pts long, overlapping by Wov points with the previous output.
        #   The similarity is calculated over the last Wsim points of output.
        #   Maximum similarity skew is Kmax pts.
        #   Each xcorr calculation is decimated by xdecim (8)
        #   The skew axis sampling is decimated by kdecim (2)
        #   Defaults (for 22k) W = 200, Wov = W/2, Kmax = 2*W, Wsim=Wov.
        #   Based on "The SOLAFS time-scale modification algorithm",
        #   Don Hejna & Bruce Musicus, BBN, July 1991.
        #   1997may16 dpwe@icsi.berkeley.edu $Header: /homes/dpwe/matlab/dpwebox/RCS/solafs.m,v 1.3 2006/04/09 20:10:20 dpwe Exp $
        #   2006-04-08: fix to predicted step size, thanks to Andreas Tsiartas
        Ss = W - Wov;
        xpts = int(len(X));  # total samples in speech signal
        ypts = int(np.round(xpts / scale_factor));  # total samples in output speech signal
        Y = np.zeros(ypts)  # output speech signl- initialize to all zeros
        # Cross-fade win is Wov pts long - it grows
        xfwin = np.arange(1, Wov + 1) / (Wov + 1);  # Start with 1
        # xfwin = np.arange(Wov)/Wov;        # Start with 0
        # Index to add to ypos to get the overlap region
        ovix = np.arange(1 - Wov, 1);
        # Index for non-overlapping bit
        newix = np.arange(1, W - Wov + 1);
        # Index for similarity chunks
        # decimate the cross-correlation
        simix = np.arange(1, Wsim + 1, xdecim) - Wsim;

        # pre-pad X for extraction
        padX = np.append(np.append(np.zeros(Wsim), X, ), np.zeros(Kmax + W - Wov));

        # Startup - just copy first bit
        Y[:Wsim] = X[:Wsim];
        xabs = 0;
        lastxpos = 0;
        lastypos = 0;
        km = 0;

        for ypos in np.arange(Wsim, ypts - W + 1, Ss):
            # Ideal X position
            xpos = int(scale_factor * ypos);
            # Overlap prediction - assume all of overlap from last copy
            kmpred = km + ((xpos - lastxpos) - (ypos - lastypos));
            lastxpos = xpos;
            lastypos = xpos;
            if (kmpred <= Kmax and kmpred >= 0):
                km = kmpred;  # no need to search
            else:
                # Calculate the skew, km by first figuring the cross-correlation
                ysim = Y[ypos + simix - 1];
                # Clear the Rxy array
                rxy = np.zeros(Kmax + 1);
                rxx = np.zeros(Kmax + 1);
                # Make sure km doesn't take us backwards
                # Kmin = np.max(0, xabs-xpos);
                Kmin = 0;
                # actually, this sounds kinda bad.  Allow backwards for now
                for k in np.arange(Kmin, Kmax + 1, kdecim):
                    xsim = padX[Wsim + xpos + k + simix - 1];
                    rxx[k] = np.linalg.norm(xsim);  ## Magnitude of vetor
                    rxy[k] = np.dot(ysim, xsim);  ## Dot product
                # Zero the pts where rxx was zero
                Rxy = np.array([]);
                for i, entry in enumerate(rxx):
                    if entry != 0.0:
                        Rxy = np.append(Rxy, rxy[i] / entry);  #### CODE CONVERSION
                    else:
                        Rxy = np.append(Rxy, 0.0);
                # Local max gives skew
                km = np.min(np.argmax(Rxy) - 1);
            xabs = xpos + km;
            # Cross-fade some points
            Y[ypos + ovix] = ((1 - xfwin) * Y[ypos + ovix]) + (xfwin * padX[Wsim + xabs + ovix - 1]);
            # Add in remaining points
            Y[ypos + newix] = padX[Wsim + xabs + newix - 1];
        return Y

    @staticmethod
    # def solafs(X, F, W=None, Wov=None, Kmax=None, Wsim=None, xdecim=None, kdecim=None):
    #
    #     #   Y = solafs(X, F, W, Wov, Kmax, Wsim, xdec, kdec)   Do SOLAFS timescale mod'n
    #     #   Y is X scaled to run F x faster.  X is added-in in windows
    #     #   W pts long, overlapping by Wov points with the previous output.
    #     #   The similarity is calculated over the last Wsim points of output.
    #     #   Maximum similarity skew is Kmax pts.
    #     #   Each xcorr calculation is decimated by xdecim (8)
    #     #   The skew axis sampling is decimated by kdecim (2)
    #     #   Defaults (for 22k) W = 200, Wov = W/2, Kmax = 2*W, Wsim=Wov.
    #     #   Based on "The SOLAFS time-scale modification algorithm",
    #     #   Don Hejna & Bruce Musicus, BBN, July 1991.
    #
    #     if W is None:
    #         W = 200
    #     if Wov is None:
    #         Wov = int(W / 2)
    #     if Kmax is None:
    #         Kmax = 2 * W
    #     if Wsim is None:
    #         Wsim = Wov
    #     if xdecim is None:
    #         xdecim = 8
    #     if kdecim is None:
    #         kdecim = 2
    #
    #     Ss = W - Wov
    #
    #     #  if(X.shape[1] = 1):
    #     # raise ValueError("solafs: X must be a single-row vector")
    #
    #     xpts = X.shape[1]
    #     ypts = round(xpts / F)
    #     Y = np.zeros((1, ypts))
    #
    #     # Cross-fade win is Wov pts long - it grows
    #     xfwin = np.arange(1, Wov + 1) / [Wov + 1]
    #
    #     # Index to add to ypos to get the overlap region
    #     ovix = np.arange(-Wov, 0, 1)
    #     # Index for non-overlapping bit
    #     newix = np.arange(0, W - Wov)
    #     # Index for similarity chunks
    #     # decimate the cross-correlation
    #     simix = np.arange(0, Wsim, xdecim) - Wsim
    #
    #     # prepad X for extraction
    #     padX = np.concatenate((np.zeros((1, Wsim)), X, np.zeros((1, Kmax + W - Wov))), axis=1)
    #
    #     # Startup - just copy first bit
    #     Y[:, 0:Wsim] = X[:, 0:Wsim]
    #
    #     xabs = 0
    #     lastxpos = 0
    #     lastypos = 0
    #     km = 0
    #     for ypos in np.arange(Wsim, ypts - W + Ss, Ss):
    #         # Ideal X position
    #         xpos = int(F * ypos)
    #         # print(['xpos=',num2str(xpos),' ypos=',num2str(ypos)]);
    #         # Overlap prediction - assume all of overlap from last copy
    #         kmpred = km + ((xpos - lastxpos) - (ypos - lastypos))
    #         lastxpos = xpos
    #         lastypos = xpos
    #         if (kmpred <= Kmax and kmpred >= 0):
    #             km = kmpred  # no need to search
    #         else:
    #             # Calculate the skew, km
    #             # .. by first figuring the cross-correlation
    #             ysim = Y[:, ypos + simix]
    #             # Clear the Rxy array
    #             rxy = np.zeros((1, Kmax + 1))
    #             rxx = np.zeros((1, Kmax + 1))
    #             # Make sure km doesn't take us backwards
    #             # Kmin = max(0, xabs-xpos);
    #             Kmin = 0
    #             # actually, this sounds kinda bad.  Allow backwards for now
    #             for k in np.arange(Kmin, Kmax + kdecim, kdecim):
    #                 xsim = padX[:, Wsim + xpos + k + simix]
    #                 rxx[0, k] = np.linalg.norm(xsim)
    #                 rxy[0, k] = np.dot(ysim, np.transpose(xsim))
    #
    #             # Zero the pts where rxx was zero
    #             Rxy = (rxx != 0) * rxy / (rxx + (rxx == 0))
    #             # Local max gives skew
    #             km = np.min(np.nonzero(Rxy[0,] == max(Rxy[0,])))
    #
    #         # km = min(find(Rxy == max(Rxy))-1)
    #
    #         xabs = xpos + km
    #         #  print(['ypos = ', int2str(ypos), ', km = ', int2str(km), '(base = ', int2str(ypos-xabs), ')'])
    #         #  fig = plt.figure()
    #         #  ax = fig.add_subplot(3,1,1)
    #         #  ax.plot(ysim)
    #         #  ax =fig.add_subplot(3,1,2)
    #         #  ax.plot(padX(Wsim + xpos + ((1-Wsim):Kmax)))
    #         #  ax =fig.subplot(3,1,3)
    #         #  ax.plot(Rxy)
             #  plt.pause()
    
             # Cross-fade some points
    #         Y[:, ypos + ovix] = ((1 - xfwin) * Y[:, ypos + ovix]) + (xfwin * padX[:, Wsim + xabs + ovix])
    #         # Add in remaining points
    #         Y[:, ypos + newix] = padX[:, Wsim + xabs + newix]
    #
    #         return Y

    # method to scaling the time in the speech file with the help of function solafs() and saving the files
    def time_scale(speech):
        # speech =self.filedata
        scale_factor = np.array([2.0, 5.0, 10.0, 20.0])
        sounds = [vsts.solafs(speech, 1 / scale) for scale in scale_factor]
        return sounds

    # method used to prepare the data from the sound file for furhter analysis
    def data_prepare(self):

        Fs = self.Fs
        filedata = self.filedata

        # FIR filter  
        f_coef = signal.firwin(201, [2 * 100 / Fs, 2 * 4000 / Fs], window='hamming', pass_zero=False)

        #        f_coef = [
        #                0.00000219494,0.00001130920,0.00031634400,-0.00013727900,0.00035698300,0.00008155700,0.00008460650,0.00045510600,-0.00009041410,0.00051753100,0.00018529400,0.00018019400,0.00067380100,-0.00004484010,0.00076004400,0.00032661200,0.00030371200,0.00098517300,-0.00000451695,0.00108642000,0.00049979100,0.00043953900,0.00138045000,0.00000269504,0.00147423000,0.00067393300,0.00054520500,0.00182497000,-0.00007892210,0.00187415000,0.00079123400,0.00055055200,0.00225898000,-0.00033180000,0.00221301000,0.00077080100,0.00036241000,0.00260439000,-0.00085848000,0.00240256000,0.00051805400,-0.00012543700,0.00277708000,-0.00177029000,0.00235343000,-0.00006132450,
        #               -0.00101837000,0.00270371000,-0.00317315000,0.00199255000,-0.00104485000,-0.00240546000,0.00234103000,-0.00515341000,0.00128179000,-0.00247673000,-0.00434556000,0.00169581000,-0.00776741000,0.00023569300,-0.00435427000,-0.00685859000,0.00084398100,-0.01103930000,-0.00106309000,-0.00661975000,-0.00992648000,-0.00004966890,-0.01497350000,-0.00244974000,-0.00915971000,-0.01350950000,-0.00070438400,-0.01959360000,-0.00365778000,-0.01181230000,-0.01759070000,-0.00066584900,-0.02503610000,-0.00427153000,-0.01438230000,-0.02228290000,0.00085782500,-0.03179530000,-0.00357045000,-0.01666220000,-0.02813630000,0.00557432000,-0.04153780000,0.00010147100,
        #               -0.01845590000,-0.03740120000,0.01893600000,-0.06130860000,0.01323650000,-0.01960260000,-0.06457640000,0.08150160000,-0.17046100000,0.16881200000,0.77988700000,0.16881200000,-0.17046100000,0.08150160000,-0.06457640000,-0.01960260000,0.01323650000,-0.06130860000,0.01893600000,-0.03740120000,-0.01845590000,0.00010147100,-0.04153780000,0.00557432000,-0.02813630000,-0.01666220000,-0.00357045000,-0.03179530000,0.00085782500,-0.02228290000,-0.01438230000,-0.00427153000,-0.02503610000,-0.00066584900,-0.01759070000,-0.01181230000,-0.00365778000,-0.01959360000,-0.00070438400,-0.01350950000,-0.00915971000,-0.00244974000,-0.01497350000,-0.00004966890,
        #               -0.00992648000,-0.00661975000,-0.00106309000,-0.01103930000,0.00084398100,-0.00685859000,-0.00435427000,0.00023569300,-0.00776741000,0.00169581000,-0.00434556000,-0.00247673000,0.00128179000,-0.00515341000,0.00234103000,-0.00240546000,-0.00104485000,0.00199255000,-0.00317315000,0.00270371000,-0.00101837000,-0.00006132450,0.00235343000,-0.00177029000,0.00277708000,-0.00012543700,0.00051805400,0.00240256000,-0.00085848000,0.00260439000,0.00036241000,0.00077080100,0.00221301000,-0.00033180000,0.00225898000,0.00055055200,0.00079123400,0.00187415000,-0.00007892210,0.00182497000,0.00054520500,0.00067393300,0.00147423000,0.00000269504,0.00138045000,
        #                0.00043953900,0.00049979100,0.00108642000,-0.00000451695,0.00098517300,0.00030371200,0.00032661200,0.00076004400,-0.00004484010,0.00067380100,0.00018019400,0.00018529400,0.00051753100,-0.00009041410,0.00045510600,0.00008460650,0.00008155700,0.00035698300,-0.00013727900,0.00031634400,0.00001130920,0.00000219494
        #        ]

        # Filter data along one-dimension with an FIR filter.
        data_raw = signal.lfilter(f_coef, 1.0, filedata)

        data_raw = data_raw[100:len(data_raw)]  # unfiltered

        len_data_raw = len(data_raw)

        data_prep = np.append(data_raw[1:len_data_raw + 1], 0)
        # data_prep = data_prep-np.transpose(data_raw)
        data_prep = data_prep - data_raw

        return data_raw, data_prep

    # function to find the related waveform data w.r.t time
    def wave_detection(self):

        data_raw, data_prep = self.data_prepare()
        #
        data_raw[self.N3 - 1: len(data_raw)] = 0  # make values after right end point zero

        # i = np.linspace(start=1, stop=filelen - N1 + 1, num=filelen - N1 + 1)

        i = np.arange(1, self.filelen - self.N1 + 2,
                      1)  # this is needed for the modern version to communicatedata upto 5 s

        i = i * (1 / self.Fs)

        timedata = np.transpose(i)

        wavedata = np.append(data_raw[self.N1 - 1:self.N2 + 1], np.zeros(self.filelen - self.N2))

        return timedata, wavedata

    def spectrogram(self):

        data_raw = self.data_prepare()[0]
        data = np.concatenate((data_raw[self.N1 - 1:self.N2], np.zeros(self.filelen - self.N2)), axis=0)
        wsize = int(np.round(29 * self.Fs / 1000))  # 29 ms window length
        o_lap = np.round(0.9 * wsize)
        # no_samples=round(29*Fs/1000); # 16 Hz of freq. resolution at fs=11025
        maxstftm = 10
        int112 = maxstftm * 10 ** -5
        # spec11 = signal.spectrogram(x = y, nfft = 512, fs = Fs,window = np.hamming(wsize),noverlap  = o_lap, mode = 'complex')
        # b1=20*np.log10((abs(spec11[2])+np.spacing(1))/int112)

        b, f, t = vsts.specgram(x=data, n=512, Fs=self.Fs, window=np.hamming(wsize), noverlap=o_lap)
#         b, f, t, im = plt.specgram(x=data, pad_to=512, Fs=self.Fs, NFFT=wsize, window=np.hamming(wsize), noverlap=o_lap)

        S = 20 * np.log10((abs(b) + np.spacing(1)) / int112)
        return S, f, t

        # function to extract pitch values from sound file using praat

    # =============================================================================
    #     def pitch_calculation(self):
    #
    #         subprocess.call(['Praat.exe', '--run', 'extract_pitch.praat', self.file, 'pitch.txt'])
    #
    #        # praatcommand = ('"Praat.exe" --run extract_pitch.praat filepath, "pitch.txt"')
    #        # os.system(praatcommand)
    #         #pitchfile = 'pitch.txt'
    #         Npitch=np.genfromtxt('pitch.txt', unpack=True)
    #
    #         t_p=Npitch[0]
    #
    #         pf=Npitch[1]
    #         #pitch_f = np.mean(pf)
    #
    #         return t_p, pf
    # =============================================================================

    # function to extract pitch values from sound file using pitch.py
    def pitch_calculation(self):

        t_p, pf = pitch_manoj.pitch_of_sound(self.filedata, self.Fs)

        return t_p, pf

    # function to generate window and it size using hamming window for pitch calculated
    def pitch_windowing(self):

        pf = self.pitch_calculation()[1]
        # find window size
        win_size = round(self.win_scale * self.Fs / np.mean(pf))
        # if window size is not equal to twice the pp then make it
        if win_size != 2 * np.ceil(win_size / 2):
            win_size = 2 * np.ceil(win_size / 2)

        win_size = int(win_size)
        window = np.hamming(int(win_size))  # apply hamming window
        return win_size, window

    # function to arrange pitch in correct order to return final pitch values and time values
    def pitch_correct_order(self):

        t_p, pf = self.pitch_calculation()
        # Atrranging the pitch in correct order... 
        # First 0.1 s pitch 0, then when unvoiced pitch 0 and from 0.9 pitch is zero

        # display the pitch between N1 and N2 only
        N1_t = self.N1 * (1 / self.Fs)  # Begining of speech
        #  N2_t = N2*(1/Fs)  # End of 1 sec of signal
        N3_t = self.N3 * (1 / self.Fs)  # end of speech

        N4_t = N3_t

        # Nt_l=N3_t-N1_t

        p_start = np.nonzero(t_p > N1_t)[0][0:1]

        p_end = np.nonzero(t_p > N4_t)[0][0:1]

        if p_start.size == 0:
            p_start = 0

        if p_end.size == 0:
            p_end = len(t_p) - 1

        t_p_plot = t_p - t_p[p_start]  # removing the bias of t_p by subtracting from tp_start

        tp_plindex = np.nonzero(t_p_plot == 0)[0]

        t_pwindow = np.round((t_p[tp_plindex + 1] - t_p[tp_plindex]) * 100)[0]

        tpp_l = np.arange(0, t_p[p_start], t_pwindow / 100)

        tpp_l_copy = tpp_l

        if (tpp_l_copy[-1] == t_p[p_start]):
            #     tpp_l_copy_1 = tpp_l_copy[:-1]
            tpp_l = tpp_l[:-1]

        tpp_l = np.append(np.transpose(tpp_l), t_p)

        # pf_l = []
        pf_l = np.zeros(len(tpp_l_copy))
        pf_l = np.append(pf_l, pf[tp_plindex[0]:p_end + 1])

        incr = 0

        # This loop works if there is a discontinuity in the pitch coming from
        # praat.. then it inserts some time points which are differed by window
        # length.. i.e 0.01

        difftpplot = []
        pf_l1 = []
        tpp_l1 = []
        llist = np.arange(0, len(tpp_l) - 1, 1)
        for item in llist:
            difftpplot = np.append(difftpplot, np.double(tpp_l[item + 1] - tpp_l[item]))
            # difftpplot[item]=np.double(tpp_l[item+1]-tpp_l[item])
            if (np.round(100 * difftpplot[item]) > t_pwindow):
                No_ofterm = int(round((100 * difftpplot[item]) / t_pwindow))
                # pf_l1[incr : incr+No_ofterm]=[0]*(incr+No_ofterm)
                pf_l1 = np.append(pf_l1, np.zeros(No_ofterm))
                # tpp_l1[incr: incr+No_ofterm]=np.arange(tpp_l[item]+(t_pwindow/100),tpp_l[item+1],(t_pwindow/100))
                tpp_l1 = np.append(tpp_l1, np.linspace(tpp_l[item] + (t_pwindow / 100), tpp_l[item + 1], No_ofterm))
                incr = incr + No_ofterm
            else:
                tpp_l1 = np.append(tpp_l1, tpp_l[item])
                pf_l1 = np.append(pf_l1, pf_l[item])
                incr = incr + 1

        tpp_l2 = tpp_l1
        pf_l2 = pf_l1
        # ExtraTimeData=np.arange(tpp_l2[-1]+t_pwindow/100,filelen/Fs + t_pwindow/100,t_pwindow/100)
        ExtraTimeData = np.arange(tpp_l2[-1] + t_pwindow / 100, self.filelen / self.Fs, t_pwindow / 100)
        tpp_l2 = np.append(tpp_l2, ExtraTimeData)  # pitch Time
        # len(np.repeat(0,len(tpp_l2)-len(pf_l2)))
        pf_l2 = np.append(pf_l2, np.repeat(0, len(tpp_l2) - len(pf_l2)))
        pf_l2 = signal.medfilt(pf_l2, 3)
        pf_final = np.double(pf_l2)  # final pitch values

        return tpp_l2, pf_final

    # function to find the area values for the jaw
    def area_calculation(self):

        data_raw, data_prep = self.data_prepare()
        win_size, window = self.pitch_windowing()

        eng_hamm = np.zeros((len(data_raw)))
        WindSigEn = np.zeros((len(data_raw)))

        EngSig = data_prep * data_prep

        # window is the hamming window
        wEn = window * window

        j = self.N1 - 1
        for i in np.arange(self.N1 - 1, self.N3, 1):
            if ((i + win_size - 1)) >= self.N3 - 1:
                break

            ww = EngSig[i:i + win_size]
            TmpWindSigEn = ww * (np.transpose(wEn))
            WindSigEn[j] = np.sum(TmpWindSigEn)

            # Calculate Rectangular Eng
            RectEn = np.sum(ww)
            # calculate energy index #
            if (RectEn > 0):
                eng_hamm[j] = WindSigEn[j] / RectEn
            else:
                eng_hamm[j] = 0
            j = j + 1

        # filter the energy valleys so that small perturbations are neglected

        f_coef_20 = signal.firwin(21, 2 * 180 / self.Fs)

        #  f_coef_20 =  [0.0061573, 0.0082247, 0.0139681, 0.0231614, 0.0351202, 0.0487674,
        #     0.0627547, 0.0756246, 0.0859905, 0.0927116, 0.0950389, 0.0927116,
        #     0.0859905, 0.0756246, 0.0627547, 0.0487674, 0.0351202, 0.0231614,
        #     0.0139681, 0.0082247, 0.0061573]

        eng_hamm_filt = signal.lfilter(f_coef_20, 1.0, eng_hamm)
        eng_hamm_filt = eng_hamm_filt[10:len(eng_hamm_filt)]
        eng_hamm_filt = np.append(eng_hamm_filt, np.zeros(len(eng_hamm) - len(eng_hamm_filt)))

        min_window_size = int((1 / 100) * self.Fs)

        tmp_index = self.N1 - 21;
        j = 0;

        sel_frames = []
        while ((tmp_index + min_window_size) < self.N3):
            arr = eng_hamm_filt[(tmp_index + 21 - 1):(tmp_index + min_window_size + 1)]
            # dvar = np.min(arr)
            tmpSelFrame = np.argmin(arr)
            tmpSelFrame = tmpSelFrame + tmp_index + 21 - 1
            if ((tmpSelFrame + win_size - 1) < self.N3):
                sel_frames = np.append(sel_frames, tmpSelFrame)
            else:
                break
            tmp_index = int(sel_frames[j])
            j = j + 1

        # reject first and last 3 frames
        sel_frames = np.array(sel_frames[3:-3], dtype=int)

        aa = []  # initialize area values to empty array
        kk = []
        a_eng = np.zeros((len(sel_frames), 12))
        s = np.zeros(self.lpc_ord + 1)
        ## Calculate the Area values only at the selected frames#####
        for i in np.arange(0, len(sel_frames), 1):
            ww = data_prep[sel_frames[i]:sel_frames[i] + win_size]
            x = ww * np.transpose(window)
            acr = vsts.XCORR_FFT(x, self.lpc_ord)  # apply cross correlation

            r = acr[(self.lpc_ord):(2 * self.lpc_ord + 2)]
            if (r[0] < 0.0001):  # zero energy signal
                kk = np.zeros(self.lpc_ord)
            else:
                # matrc, cc=ac2rc(r) # reflection coefficients using durbin algorithm
                matrc = vsts.levinson(r)
                # matrc=linalg.solve_toeplitz(r)
                kk = matrc[0:self.lpc_ord]
            # print(i)
            # print(matrc)

            # TO COMPUTE VT AREA
            if (any(kk) == 0):  # all elements of kk are zero...zero energy...area also zero
                # s = np.append(s, 0)
                s[self.lpc_ord] = 0  # careful length(s) is error prone changes dynamically
            else:
                # s = np.append(s, 1)
                s[self.lpc_ord] = 1  # initialization of area s[m] = 1 (m=lpc_ord)

            for m in np.arange(self.lpc_ord - 1, -1, -1):
                s[m] = ((1 + kk[m]) / (1 - kk[m])) * s[m + 1]
                # observe for m=lpc_ord [m+1]=1 is used

            # print(s)
            a_eng[i,] = s[0:self.lpc_ord]

            ## calculating the area values with 5ms shift obtained using the energy
            # window selected area values and weighted average

        # enind = 1

        for i in np.arange(self.N1, self.N3 + 1, self.wshift_samples):
            # To find window size
            if i + win_size - 1 > self.N3:
                # if ((i+win_size-1)) > N2
                break;

            sl_frame = np.nonzero(sel_frames > i)[0]

            # if isempty(sl_frame): # if beyond the range use last p_freq
            if np.all(sl_frame == 0):
                ww = data_prep[i - 1:i + win_size - 1]
                # ww=data_prep[sel_frames[i]:sel_frames[i]+win_size]

                x = ww * np.transpose(window)

                acr = vsts.XCORR_FFT(x, self.lpc_ord)
                r = acr[(self.lpc_ord):(2 * self.lpc_ord + 2)]
                if (r[0] < 0.0001):  # zero energy signal
                    kk = np.zeros(self.lpc_ord)
                else:
                    # matrc, cc=ac2rc(r) # reflection coefficients using durbin algorithm
                    matrc = vsts.levinson(r)
                    # matrc  = linalg.solve_toeplitz(r)
                    kk = matrc[0:self.lpc_ord + 1]

                if (all(kk) == 0):  # all elements of kk are zero...zero energy...area also zero
                    s = np.append(s, 0)
                    # s[lpc_ord]=0 #careful length(s) is error prone changes dynamically
                else:
                    s = np.append(s, 1)
                    # s[lpc_ord]=1 # initialization of area s(m+1) = 1 (m=lpc_ord)

                for m in np.arange(self.lpc_ord - 1, -1, -1):
                    s[m] = ((1 + kk[m]) / (1 - kk[m])) * s[m + 1]
                    # observe for m=lpc_ord s(m+1)=1 is used

                if (len(aa) > 0):
                    aa = np.append(aa, 0.5 * s[0:self.lpc_ord] + 0.5 * aa[-(self.lpc_ord + 2),])
                    # a[len(a):(len(a)+lpc_ord)] = 0.5*s[0:lpc_ord]+0.5*a[-(lpc_ord+2),] #wastage of memory
                else:
                    aa[0:self.lpc_ord] = s[0:self.lpc_ord]

            else:
                if sl_frame[0] > 1:
                    tmp_total = sel_frames[sl_frame[0]] - sel_frames[sl_frame[0] - 1] + 1
                    A1 = (sel_frames[sl_frame[0]] - i) / tmp_total  # weights calculated from the distance
                    B1 = (i - sel_frames[sl_frame[0] - 1]) / tmp_total
                    aa = np.append(aa, B1 * a_eng[sl_frame[0],] + A1 * a_eng[sl_frame[0] - 1,])

                else:
                    if (len(aa) > 0):
                        aa = np.append(aa, np.zeros(12))
                        # a[len(a):(len(a)+lpc_ord)] = np.zeros(1,12) #wastage of memory
                    else:
                        aa[1:self.lpc_ord] = np.zeros(12)

        return aa, WindSigEn

    # function to find the area intensity w.r.t time
    def area_intensity(self):
        aa, WindSigEn = self.area_calculation()

        av_en = WindSigEn[np.arange(self.N1 - 1, self.N3, self.wshift_samples)]

        # axes(handles.axes9);    ##pitch and intensity display

        No_samples = np.arange(self.N1, self.N2, self.wshift_samples)  # should be in the order of  N/5 ms shift

        t_en = np.arange(1, len(No_samples), 1) * self.wshift_time_ms * 0.001
        t_en = np.append(0, t_en)
        av_enful = av_en

        ## Energy to dB conversion taking 100 dB as maximum energy
        for intr in np.arange(0, len(av_enful), 1):
            if (av_enful[intr] <= 1e-3):
                av_enful[intr] = 0

        for intr1 in np.arange(len(av_enful) - 1, 0, -1):
            if (av_enful[intr1] >= 1e-3):
                # endpt11=intr1
                break

        # max12m=max(av_enful)
        max12m = 16
        int01 = max12m * 10 ** -5
        new_avg_en = 20 * np.log10((av_enful / int01) + np.spacing(1))

        # Range display for 0 to 90 dB
        more80 = np.nonzero(new_avg_en < 0)[0]
        new_avg_en[more80] = 0

        ## Appending additional zero due to error caused by filtering in the first
        new_avg_en = np.append(new_avg_en, np.zeros(int((self.filelen / (self.Fs * 5e-3)) - len(new_avg_en))))
        # new_avg_en[len(new_avg_en)+1: round(filelen/(Fs*5e-3))]=0

        t_en = np.append(t_en, np.arange(1000 * (t_en[-1] + 5e-3) / 1000, (self.filelen / self.Fs) - 5e-3,
                                         5e-3))  # intensity(energy) level
        # t_en[len(t_en)+1: round(filelen/(Fs*5e-3))]=round(np.arange(1000*(t_en[-1]+5e-3))/1000),((filelen/Fs)-5e-3, 5e-3)

        av_enful1 = np.transpose(new_avg_en)  # intensity(enery) values

        return av_enful1, t_en

    # function to find the areagram matrix
    def areagram(self):

        aa = self.area_calculation()[0]

        lpc_ord = self.lpc_ord
        Fs = self.Fs
        filelen = self.filelen
        ag_a = np.sqrt(aa)
        ag_x = np.arange(1, 13, 1)
        # ag_z=np.arange(0,(len(a)/lpc_ord),1)
        ag_q = ag_a.reshape(int(len(aa) / lpc_ord), lpc_ord)

        # compensation for error due to filtering in the early stage
        ag_q = np.append(ag_q, np.zeros((int(filelen / (Fs * 5e-3) - (len(aa) / lpc_ord)), len(ag_q[0]))), axis=0)

        # ag_z=np.arange(0,round(filelen/(Fs*5e-3)),1) #As we know  frames will be placed at every 5ms

        ag_x_spl = np.linspace(1, lpc_ord, 41)  # areagram_GLdistance

        ag_t_spl = np.linspace(0, filelen / Fs, round(filelen / (Fs * 5e-3)))  # areagram_time

        # ag_q_spl = interpolate.spline(ag_x,ag_q,ag_x_spl)  # areagram matrix
        ag_q_spl = interpolate.interp1d(ag_x, ag_q)(ag_x_spl)  # areagram matrix
        ag_q_spl = np.transpose(ag_q_spl)

        return ag_q_spl, ag_t_spl, ag_x_spl

    # functio to apply spline interpolation  to the area caluclated
    def area_spline(self):

        aa = self.area_calculation()[0]
        lpc_ord = self.lpc_ord
        Fs = self.Fs
        filelen = self.filelen

        aa = np.sqrt(aa)

        # LIMIT ALL THE AREA VALUES TO 7.8 SO AS TO SCALING TO IMAGE WORK PROPERLY
        # So any values larger than 7.8 will be saturated to 7.8

        aa[np.nonzero(aa > 7.8)] = 7.8

        aa = aa.reshape(int(len(aa) / lpc_ord), lpc_ord)  # (12 X Nsegments)
        aa = np.transpose(aa)

        # Adjustments due to filtering error
        aa = np.append(aa, np.zeros((len(aa), int(filelen / (Fs * 5e-3) - len(aa[0])))), axis=1)

        # Need 20 section area values for display_l and hence interpolate the given
        # 12 values to 20 area values to fit b-spline
        disp_ag_x = np.arange(1, 13)
        # disp_ag_x_spl = np.linspace(1, lpc_ord, 20)
        # disp_ag_x_spl=np.arange(1,lpc_ord+(lpc_ord-1)/19,(lpc_ord-1)/19)

        a_tran = np.transpose(aa)
        disp_ag_q_spl = np.zeros((len(aa[0]), 20))
        knots = 4
        nSplinePts = 20
        for i in np.arange(0, len(aa[0]), 1):
            a_t = a_tran[i,]
            disp_matx, disp_maty = vsts.fit_LSBSpline(disp_ag_x, a_t, knots, nSplinePts)
            disp_ag_q_spl[i,] = disp_maty

        disp_dist = disp_ag_q_spl

        return disp_dist

    # function to display jaw matrix and POA matrix for animation
    def area_animation(self):

        disp_dist = self.area_spline()

        # get and plot the base figure in the erase mode
        h1 = [40, 53, 64, 76, 87, 100, 113, 131, 151, 170, 188, 200, 215, 223, 227, 229, 231, 230, 229]
        k1 = [271, 267, 262, 253, 246, 237, 228, 221, 220, 220, 224, 232, 244, 262, 271, 284, 299, 311, 326]
        h2 = [42, 55, 72, 88, 101, 112, 124, 136, 148, 161, 172, 182, 192, 200, 205, 206, 205, 203, 202]
        k2 = [278, 282, 276, 274, 270, 266, 260, 258, 255, 255, 256, 259, 266, 277, 284, 293, 306, 314, 325]

        jaw_rest_h = np.array([45, 38, 32, 31, 36, 42, 40, 39, 34, 35, 41, 51, 66, 147, ])

        jaw_rest_k = np.array([278, 277, 278, 287, 293, 300, 308, 321, 336, 346, 350, 354, 353, 350])
        # original one
        glottal_h = np.array([161, 170, 174, 175])

        glottal_k = np.array([357, 375, 399, 406])

        jaw_length1 = len(jaw_rest_h)

        jaw_length = len(jaw_rest_h)

        mslope = np.array(
            [1.6250, 1.7500, 1.7143, 2.4167, 2.9091, 7.4000, 11.6667, 3.8889, 2.0000, 1.5000, 0.9565, 0.6522, 0.5909,
             0.3913, 0.2692, 0.1111, 0.0370, 0.0690, 0.1379, 0.2759])

        # plot the jaw fist and remeber the handle for the same

        # plt.plot(jaw_rest_h,jaw_rest_k)
        # plt.show()

        # segsh=plot(h2,k2,'color','k','LineWidth',5,'EraseMode','xor');
        seg_length = len(h2)
        #  maxk2 = np.min(k2)
        maxk2ind = np.argmin(k2)

        upperLine2Spline = np.array(
            [[40, 271], [52.9360004286472, 266.875809062708], [65.7326839607545, 260.395195908744],
             [77.7510094891844, 251.996672314888], [88.9972788100407, 244.530499768412],
             [101.560947818378, 235.919343818046], [116.692152159854, 226.564163048946],
             [133.708040851485, 221.156663555356], [149.914887715874, 220.205821658707],
             [167.892918504593, 220.514554160563], [186.490022596768, 223.782507702690],
             [203.001246645247, 234.400997316198], [215.864734030133, 248.010750852725],
             [223.677580897805, 263.524557020062], [228.340120595860, 280.394533140559],
             [230.657204479494, 296.905521961607]])
        lowerLine2Spline = np.array(
            [[42, 278], [57.9927931600524, 280.254565293355], [72.4275149051325, 276.406052700154],
             [87.4654902081469, 273.964889725696], [102.635925807367, 269.308349627955],
             [115.665455111959, 264.016517163537], [126.454097514820, 260.146358766186],
             [137.424925350213, 257.463405611480], [150.097978451390, 255.440051443791],
             [161.336188825983, 255.251769623771], [171.469861790117, 256.478571860473],
             [180.639420766243, 259.366302226416], [189.352839974387, 264.788981515045],
             [197.529839155633, 273.620755945127], [203.395452733142, 283.808710834683],
             [205.513110321672, 296.098490176520]])

        h2 = np.append(np.transpose(lowerLine2Spline[:, 0]), h2[len(h2) - 1])
        k2 = np.append(np.transpose(lowerLine2Spline[:, 1]), k2[len(k2) - 1])

        h2[0] = jaw_rest_h[0]
        k2[0] = jaw_rest_k[0]

        # str=0
        seg_length = len(h2);
        h2_f = []
        k2_f = []
        for rev in np.arange(len(h2) - 2, -1, -1):
            h2_f = np.append(h2_f, h2[rev])
            k2_f = np.append(k2_f, k2[rev])
        # str=str+1

        rx = np.concatenate((jaw_rest_h, glottal_h, h2_f), axis=0)
        ry = np.concatenate((jaw_rest_k, glottal_k, k2_f), axis=0)
        rx = rx[0:jaw_length + seg_length - 1 + len(glottal_h)]
        ry = ry[0:jaw_length + seg_length - 1 + len(glottal_h)]

        # dmax corresponding maximum area from the observations for each section
        # from wakita algorithm
        dmax = 7.8

        # dmaxgraph = sqrt((y2- y1)2 + (x2-x1)2) corresponds to maximum distance from the graph
        dmaxgraph = 50  # as used for animation 'a' in animation package

        mslope = (upperLine2Spline[:, 1] - lowerLine2Spline[:, 1]) / (upperLine2Spline[:, 0] - lowerLine2Spline[:, 0])
        d_top_rest = []
        d_top_rest = np.append(d_top_rest, np.sqrt(((upperLine2Spline[0, 1] - lowerLine2Spline[0, 1])) ** 2 + (
        (upperLine2Spline[0, 0] - lowerLine2Spline[0, 0])) ** 2))
        d_top_rest = np.append(d_top_rest, np.sqrt(((upperLine2Spline[1, 1] - lowerLine2Spline[1, 1])) ** 2 + (
        (upperLine2Spline[1, 0] - lowerLine2Spline[1, 0])) ** 2))

        segmentnum = 0

        sd = np.size(disp_dist, 0)

        h1 = upperLine2Spline[:, 0]
        k1 = upperLine2Spline[:, 1]

        mat_jaw_new_hr = np.zeros((sd, len(jaw_rest_h)))
        mat_jaw_new_kr = np.zeros((sd, len(jaw_rest_h)))
        mat_k2trng = np.zeros((sd, 3))
        mat_h2trng = np.zeros((sd, 3))
        mat_px = np.zeros((sd, len(jaw_rest_h) + len(glottal_h) + 16))
        mat_py = np.zeros((sd, len(jaw_rest_h) + len(glottal_h) + 16))
        mat_h2 = np.zeros((sd, len(h2)))
        mat_k2 = np.zeros((sd, len(k2)))
        mat_px_trng = np.zeros((sd, len(jaw_rest_h) + len(glottal_h) + 3))
        mat_py_trng = np.zeros((sd, len(jaw_rest_h) + len(glottal_h) + 3))

        for segmentnum in np.arange(0, sd):
            # axes(handles.axes4);
            # obtain the actual d's from the wakita algorithm
            d_top_a = disp_dist[segmentnum, :]

            # calculation of dnew on the graph.normalized value
            d_top_a = d_top_a * (dmaxgraph / dmax)
            temp = ((d_top_a[0] - d_top_rest[0]) / np.sqrt(1 + (mslope[0] * mslope[0])))

            if (mslope[0] < 0):
                jaw_new_hr = jaw_rest_h[0:(jaw_length1)] - temp
                jaw_new_kr = jaw_rest_k[0:(jaw_length1)] - mslope[0] * temp
            else:
                jaw_new_hr = jaw_rest_h[0:(jaw_length1)] + temp
                jaw_new_kr = jaw_rest_k[0:(jaw_length1)] + mslope[0] * temp

            jaw_new_hr[jaw_length1 - 1] = jaw_rest_h[jaw_length1 - 1]
            jaw_new_kr[jaw_length1 - 1] = jaw_rest_k[jaw_length1 - 1]

            h2[0] = jaw_new_hr[0]
            k2[0] = jaw_new_kr[0]

            # dmin=100    #initialising to max possible /big invalid value, to be %%
            # reiteratively stored by lower encountered values%%

            # what about negative distances ?. Taken care by the d_top_a - d_top_rest logic
            for i in np.arange(1, len(h1)):
                temp = ((d_top_a[i]) / np.sqrt(1 + (mslope[i] * mslope[i])))
                #  if dmin> d_top_a()

                if (mslope[i] < 0):
                    h2[i] = h1[i] - temp
                    k2[i] = k1[i] - (mslope[i]) * temp
                else:
                    if (upperLine2Spline[i, 1] < lowerLine2Spline[i, 1]):
                        h2[i] = h1[i] + temp;
                        k2[i] = k1[i] + (mslope[i]) * temp
                    else:
                        h2[i] = h1[i] - temp;
                        k2[i] = k1[i] - (mslope[i]) * temp

            ##For calculating mid vertex (place of maximum constriction) of triangle 
            # maxk2 = np.min(disp_dist[segmentnum,1:12])
            maxk2ind = np.argmin(disp_dist[segmentnum, 1:12])

            # draw a triangle

            h2trng = np.array((h2[0], h2[maxk2ind], 202.0))  # this restricts the end points of the animation on
            k2trng = np.array((k2[0], k2[maxk2ind], 321.0))  # so that only portion above velum is animated

            str = 1;
            h2trng_f = []
            k2trng_f = []
            for rev in np.arange(2, -1, -1):
                h2trng_f = np.append(h2trng_f, h2trng[rev])
                k2trng_f = np.append(k2trng_f, k2trng[rev])
                str = str + 1

            px_trng = np.concatenate((jaw_new_hr, glottal_h, h2trng_f), axis=0)
            py_trng = np.concatenate((jaw_new_kr, glottal_k, k2trng_f), axis=0)

            str = 1
            h2_f = []
            k2_f = []
            for rev in np.arange(15, -1, -1):
                h2_f = np.append(h2_f, h2[rev])
                k2_f = np.append(k2_f, k2[rev])
                str = str + 1

            px = np.concatenate((jaw_new_hr, glottal_h, h2_f), axis=0)
            py = np.concatenate((jaw_new_kr, glottal_k, k2_f), axis=0)

            #### storing all variables needed for generating image for all frames
            #########index i###### 

            mat_jaw_new_hr[segmentnum] = jaw_new_hr
            mat_jaw_new_kr[segmentnum] = jaw_new_kr
            mat_k2trng[segmentnum] = k2trng
            mat_h2trng[segmentnum] = h2trng
            mat_px[segmentnum] = px  # jawdata - x cordinate
            mat_py[segmentnum] = py  # jawdata - y cordinate
            mat_h2[segmentnum] = h2
            mat_k2[segmentnum] = k2
            mat_px_trng[segmentnum] = px_trng
            mat_py_trng[segmentnum] = py_trng

        ## matrix for animation
        K2_POA_pos = mat_k2trng[:, 1]
        h2_POA_pos = mat_h2trng[:, 1]

        return mat_px, mat_py, K2_POA_pos, h2_POA_pos

    ## Plot and save wave data
    def plot_waveform(self):

        timedata, wavedata = self.wave_detection()
        np.savetxt(os.path.join('.//output//', 'waveform_values.txt'), wavedata)  ## Y axix
        np.savetxt(os.path.join('.//output//', 'waveform_time.txt'), timedata)  ## X axis
        fig1, ax1 = plt.subplots(1, 1)
        ax1.plot(timedata, wavedata)
        ax1.set(title='waveform', ylabel='Sig. (norm)', xlabel='Time(s)')
        plt.show()

    ## Plot and save Pitch data
    def plot_pitch(self):

        pitchTime, pitchVal = self.pitch_calculation()
        # pitchTime, pitchVal  = self.pitch_correct_order()

        np.savetxt(os.path.join('.//output//', 'pitch_time.txt'), pitchTime)
        np.savetxt(os.path.join('.//output//', 'pitch_values.txt'), pitchVal)
        fig2, ax2 = plt.subplots(1, 1)
        ax2.plot(pitchTime, pitchVal)
        ax2.set(title='Pitch_Plot', ylabel='Pitch (Hz)', xlabel='Time(s)')
        plt.show()

    ## Plot and save Intensity data 

    def plot_energy(self):

        energyVal, energyLevel = self.area_intensity()
        np.savetxt(os.path.join('.//output//', 'intensity_time.txt'), energyLevel)
        np.savetxt(os.path.join('.//output//', 'intensity_values.txt'), energyVal)
        fig3, ax3 = plt.subplots(1, 1)
        ax3.plot(energyLevel, energyVal)
        ax3.set(title='Energy Plot', ylabel='Level (dB)', xlabel='Time(s)')
        plt.show()

    ## Plot and save area data
    def plot_areagram(self):

        ag_matrix, ag_time, ag_gldistance = self.areagram()
        # np.savetxt(os.path.join('.//output//', 'areagram_time.txt'), ag_time)
        # np.savetxt(os.path.join('.//output//', 'areagram_GLdistance.txt'), ag_gldistance)
        # np.savetxt(os.path.join('.//output//', 'areagram_matrix.txt'), ag_matrix)
        # plt.figure(figsize=(8,15)) # 8 is width, 15 is height
        fig4, ax4 = plt.subplots(1, 1)
        ax4.imshow(ag_matrix, origin='lower')
        # plt.colorbar(use_gridspec=True)
        ax4.set(title='Areagram')
        # plt.ylabel('Freq. (norm)')
        plt.show()

    ## Plot and save Spectrogram data 
    def plot_spectrogram(self):

        S_matrix, freq, t = self.spectrogram()
        # np.savetxt(os.path.join('.//output//', 'spectrogram_matrix.txt'), S_matrix)
        # np.savetxt(os.path.join('.//output//', 'spectrogram_frequency.txt'), freq)
        # np.savetxt(os.path.join('.//output//', 'spectrogram_time.txt'), t)
        # plt.figure(figsize=(8,15)) # 8 is width, 15 is height
        fig5, ax5 = plt.subplots(1, 1)
        ax5.imshow(S_matrix, origin='lower')
        # fig5.colorbar(use_gridspec=True)
        # ax5.set(ylim = [min(time), max(time)])
        ax5.set(title='Spectrogram')
        ax5.set(ylabel="frequency [Norm]")
        plt.show()

    ## save jaw area data 
    def plot_area_animation(self):

        mat_px, mat_py, K2_POA_pos, h2_POA_pos = self.area_animation()

        # np.savetxt(os.path.join('.//output//','mat_py.txt'),mat_py)
        # np.savetxt(os.path.join('.//output//','mat_px.txt'),mat_px)

        ## save POA data 
        # np.savetxt(os.path.join('.//output//','mat_px_POA.txt'),h2_POA_pos)
        # np.savetxt(os.path.join('.//output//','mat_py_POA.txt'),K2_POA_pos)
        img1 = plt.imread("images/Female_skp.png")
        img2 = plt.imread("images/Male_skp.png")
        fig6, ax6 = plt.subplots(1, 2)
        os.system("aplay self.file")
        for i in range(len(mat_px)):
            # plt.clf()
            ax6[0].cla()
            ax6[0].imshow(img1)
            # ax1.set(ylim=[-425,-200])
            ax6[0].plot(-mat_px[i] + 425, mat_py[i], color='black')
            ax6[0].fill_between(-mat_px[i] + 425, 0, mat_py[i], facecolor=[(254 / 255, 157 / 255, 111 / 255)])
            # ax1[0].fill_between(-mat_px[i]+425,0,mat_py[i])
            ax6[0].scatter(-h2_POA_pos[i] + 425, K2_POA_pos[i], color='red', s=150)
            ax6[0].axis("off")

            ax6[1].cla()
            ax6[1].imshow(img2)
            # ax1.set(ylim=[-425,-200])
            ax6[1].plot(mat_px[i], mat_py[i], color='black')
            ax6[1].fill_between(mat_px[i], 0, mat_py[i], facecolor=[(254 / 255, 157 / 255, 111 / 255)])
            # ax1[1].fill_between(mat_px[i],0,mat_py[i])
            ax6[1].scatter(h2_POA_pos[i], K2_POA_pos[i], color='red', s=150)
            ax6[1].axis("off")
            plt.pause(.001)

    def plot_area_animation2(self):
        img2 = plt.imread("images/male.png")
        mat_px, mat_py, K2_POA_pos, h2_POA_pos = self.area_animation()
        import matplotlib.animation as animation
        fig, ax = plt.subplots()
        x = []
        y = []

        jaw_outline, = ax.plot(x, y, color='black')
        img = ax.imshow(img2)

        def init():  # only required for blitting to give a clean slate.
            x = mat_px[0]
            y = mat_py[0]
            jaw_outline.set_data(x, y)
            return img, jaw_outline

        def animate(i):
            # update the data
            x = mat_px[i]
            y = mat_py[i]
            jaw_outline.set_data(x, y)
            poa = plt.scatter(h2_POA_pos[i], K2_POA_pos[i], color='red', s=150)
            jaw_area_fill = plt.fill_between(x, y, 0, facecolor=[(254 / 255, 157 / 255, 111 / 255)])

            return img, jaw_outline, jaw_area_fill, poa

        ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(mat_px), interval=0.1, blit=True)

    # start_time = time.time()
    # print("Execution time  : %s seconds " % (time.time() - start_time))
