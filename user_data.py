# -*- coding: utf-8 -*-
"""
@author: AnuragSharma

This file defines a class to hold data for a user. The user can be a trainee or
a trainer using the VSTS software.
"""

import numpy as np
import sounddevice as sd


class UserData(object):

    def __init__(self, user_name):
        """
        Currently the only value needed to initialize the user data is user name.
        """
        self.user_name = user_name

        # User audio signal (recorded speech) data
        # Sampling rate is hard-coded to 22050
        # Signal recording length is hard-coded to 10s
        self.signal_fs = 10000
        self.signal_duration = 10
        num_samples = self.signal_duration * self.signal_fs + 1
        self.signal_data = np.zeros(num_samples)  # y
        self.signal_time = np.linspace(0.0, self.signal_duration, num_samples)  # x
        self.x_c = None
        self.y_c = None

    def record_signal(self, duration, fs):
        """
        Given signal data and sampling frequency, infer the signal duration and
        set the signal time data
        """
        self.signal_fs = fs
        self.signal_duration = duration
        num_samples = (duration * fs + 1)
        self.signal_time = np.linspace(0.0, self.signal_duration, num_samples)
        self.signal_data = sd.rec(int(num_samples), samplerate=fs, channels=1)
        sd.wait()

    def set_signal_data(self, data):
        """
        Given signal data and sampling frequency, infer the signal duration and
        set the signal time data
        """
        self.signal_data = data
        num_samples = len(data)
        self.signal_duration = num_samples / self.signal_fs
        self.signal_time = np.linspace(0.0, self.signal_duration, num_samples)

    def get_signal_data(self):
        """
        Return the following signal values (used for plotting)
        :return:
        self.signal (speech signal sampled at 22050 Hz for 10s)
        self.time (time stamp values for each signal sample)
        """
        return self.signal_time, self.signal_data

    def set_area_spectro_area_data(self, r_d, s_x, s_f,s_t, a_x,a_t, a_f):
        """setting values derived from areagram()[0] function"""
        self.raw_data = r_d
        self.spectrogram_x = s_x
        self.spectrogram_f = s_f
        self.spectrogram_t = s_t
        self.areagram_x = a_x
        self.areagram_t = a_t
        self.areagram_f = a_f

    def set_animation_data(self, mat_px, mat_py, K2_POA_pos, h2_POA_pos):
        """setting values derived from areagram()[0] function"""
        self.mat_px = mat_px
        self.mat_py = mat_py
        self.K2_POA_pos = K2_POA_pos
        self.h2_POA_pos = h2_POA_pos


    def get_animation_data(self):
        return self.mat_px, self.mat_py, self.K2_POA_pos, self.h2_POA_pos

    def set_animation_data_s_op2(self, mat_px, mat_py, K2_POA_pos, h2_POA_pos):
        """setting values derived from areagram()[0] function"""
        self.mat_px_s_op2 = mat_px
        self.mat_py_s_op2 = mat_py
        self.K2_POA_pos_s_op2 = K2_POA_pos
        self.h2_POA_pos_s_op2 = h2_POA_pos

    def get_animation_data_s_op2(self):
        return self.mat_px_s_op2, self.mat_py_s_op2, self.K2_POA_pos_s_op2, self.h2_POA_pos_s_op2


    def set_flag(self,file_change_flag):
        self.file_change_flag =file_change_flag

    def get_flag(self):
        """getting initial and final values on graphs"""
        return self.file_change_flag

if __name__ == '__main__':
    test_user_data = UserData('Test1')
    print(test_user_data.user_name)
    print(test_user_data.signal_duration)
    print(test_user_data.signal_fs)
    print(test_user_data.signal_data)
    print(test_user_data.signal_time)
