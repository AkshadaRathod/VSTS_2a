"""
Created on Fri Jan 3 11:34:40 2019

@author: Aksahda Govind rathod
"""
import os

import wx
import matplotlib
from matplotlib import ticker

matplotlib.use('WXAgg')

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


class Analysis_Lower_Left(wx.Panel):
    def __init__(self, parent, user_data):
        """ Initialize everything here """
        super(Analysis_Lower_Left, self).__init__(parent, style=wx.BORDER_DOUBLE)
        self.SetBackgroundColour(wx.Colour("White"))
        self.x_vals = None
        self.y_vals = None
        self.user_data = user_data
        self.colorbar_area = None
        self.InitUI()

    def InitUI(self):
        self.main_box = wx.BoxSizer(wx.VERTICAL)
        self.x, self.y = None, None

        self.xycordi = wx.StaticText(self, -1, " ")
        self.main_box.Add(self.xycordi, 0, wx.EXPAND, 1)

        spectrogram_box = wx.BoxSizer(wx.VERTICAL)
        self.figure_spectrogram = Figure(figsize=(2, 2))

        self.axes_spectrogram = self.figure_spectrogram.add_subplot(111)

        # self.figure_spectrogram.subplots_adjust(right=1.055)
        self.canvas_spectrogram = FigureCanvas(self, -1, self.figure_spectrogram)
        # self.canvas_spectrogram.mpl_connect('motion_notify_event', self.format_coord)
        # self.canvas_spectrogram.mpl_connect('motion_notify_event', self.mouse_move)

        self.axes_spectrogram.set_ylabel('Freq.\n(norm)', fontname="Arial", fontsize=10)
        self.figure_spectrogram.canvas.mpl_connect('button_press_event', self.on_spectrogram_dclick)

        spectrogram_box.Add(self.canvas_spectrogram, 0, wx.EXPAND)
        self.main_box.Add(spectrogram_box, 0, wx.EXPAND, 1)

        areagram_box = wx.BoxSizer(wx.VERTICAL)
        self.figure_areagram = Figure(figsize=(2, 2))
        # self.figure_areagram.clf() #clear fig 25_02

        # self.figure_areagram.subplots_adjust(right=1.055)
        self.axes_areagram = self.figure_areagram.add_subplot(111)

        self.canvas_areagram = FigureCanvas(self, -1, self.figure_areagram)
        self.axes_areagram.set_ylabel('Dist. (norm)', fontname="Arial", fontsize=10)
        self.figure_areagram.canvas.mpl_connect('button_press_event', self.on_areagram_dclick)
        areagram_box.Add(self.canvas_areagram, 0, wx.EXPAND)
        self.main_box.Add(areagram_box, 0, wx.EXPAND, 1)

        self.statusbar_sizer = wx.BoxSizer(wx.VERTICAL)
        self.statusbar = wx.StatusBar(id=1, name='statusBar1', parent=self,
                                       style=0)
        # self.statusbar = wx.StatusBar(self, 1)
        # self.statusbar.SetStatusWidths(100)
        self.statusbar.SetStatusText('X =')
        self.statusbar_sizer.Add(self.statusbar)
        self.main_box.Add(self.statusbar_sizer)

        self.main_box.AddSpacer(80)
        self.SetSizer(self.main_box)

        # cordi_box = wx.BoxSizer(wx.HORIZONTAL)
        # self.xycordi = wx.StaticText(self, -1, "x= "+str(self.x)+"  y="+str(self.y))

        # cordi_box.Add(self.xycordi, 0, wx.EXPAND)
        # self.main_box.Add(cordi_box, 0, wx.EXPAND | wx.ALIGN_CENTER)

        # self.statusbar = wx.Frame.CreateStatusBar(1)

    def UpdateStatusBar(self, event):

        # if event.inaxes and event.inaxes.get_navigate():
        #     s = event.inaxes.format_coord(event.xdata, event.ydata)
        #
        #     cordi_box = wx.BoxSizer(wx.HORIZONTAL)
        #     self.xycordi = wx.StaticText(self, -1,
        #                                  "val= " +s)
        #     font = wx.Font(8, wx.DECORATIVE, wx.NORMAL, wx.BOLD)
        #     self.xycordi.SetFont(font)
        #     cordi_box.Add(self.xycordi, 0, wx.EXPAND)
        #     self.main_box.Add(cordi_box, 0, wx.EXPAND | wx.ALIGN_CENTER)

        if event.inaxes:

            cordi_box = wx.BoxSizer(wx.HORIZONTAL)
            # event.zdata = self.s_matrix
            # self.axes_spectrogram.format_coord = "text_string_made_from({:.2f},{:.2f})".format(self.x, self.y)
            self.xycordi.SetLabel("X (Time)= " + str(round(event.xdata, 3)) + ",  Y (Frequency)=" + str(
                                             round(event.ydata, 3)) + ", Z  (Spectrum Level)=" + str(round(self.spec_img.get_cursor_data(event), 3)))

            print()
            font = wx.Font(8, wx.DECORATIVE, wx.NORMAL, wx.BOLD)
            self.xycordi.SetFont(font)
            cordi_box.Add(self.xycordi, 0, wx.EXPAND)
            self.main_box.Add(cordi_box, 0, wx.EXPAND | wx.ALIGN_CENTER)

    def draw_spectrogram_data(self):
        """Draw the object data into an XY plot and refresh the plot"""
        self.axes_spectrogram.clear()
        # spectro_bar = self.axes_spectrogram.imshow(self.s_matrix, origin='lower', cmap = "gray",aspect='auto')
        # self.axes_spectrogram.pcolormesh(self.spectro_t, self.spectro_f / 10000, self.s_matrix,cmap='gist_yarg')

        data_raw = self.raw_data
        N1 = 1
        N2 = len(data_raw) - 100
        data = np.concatenate((data_raw[N1 - 1:N2], np.zeros(len(data_raw) - N2)), axis=0)
        wsize = 290  # 29 ms window length
        o_lap = np.round(0.9 * wsize)
        self.spec_mat, self.spec_f, self.spec_t, self.spec_img = self.axes_spectrogram.specgram(x=data, pad_to=512, NFFT=wsize,
                                                                            window=np.hamming(wsize), Fs=10000,
                                                                            noverlap=o_lap, cmap='gist_yarg')
        self.axes_spectrogram.xaxis.set_major_locator(ticker.LinearLocator(6))

        # cbar = self.figure_spectrogram.colorbar(spectro_bar, ax=self.axes_spectrogram, orientation='vertical', pad=0.02)
        # cbar.ax.tick_params(labelsize=7)
        self.axes_spectrogram.tick_params(axis='both', which='major', labelsize=8)
        self.axes_spectrogram.set_ylabel('Freq. (norm)', fontname="Arial", fontsize=10, labelpad=10)
        self.numrows, self.numcols = self.spec_mat.shape

        self.x  = self.spec_t
        self.y  = self.spec_f
        self.z  = self.spec_mat

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        # self.axes_spectrogram.format_coord = self.format_coord
        self.canvas_spectrogram.mpl_connect('motion_notify_event', self.mouse_move)
        self.canvas_spectrogram.draw()
        self.canvas_spectrogram.Refresh()

    def set_spectrogram_data_plot(self, r_d, start_time, end_time):
        """To set the data coming from main window for plotting """
        self.raw_data = r_d
        self.start_time = start_time
        self.end_time = end_time
        self.draw_spectrogram_data()


    def mouse_move(self, event):
        if not event.inaxes:
            return
        x, y = event.xdata, event.ydata
        col = int(x / self.dx + 0.5)
        row = int(y / self.dy + 0.5)
        xyz_str = ''
        if ((col >= 0) and (col < self.numcols) and (row >= 0) and (row < self.numrows)):
            z = self.spec_mat[row, col]
            xyz_str = 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
            print(xyz_str)
            return xyz_str
        else:
            xyz_str = 'x=%1.4f, y=%1.4f' % (x, y)
            print(xyz_str)
            return xyz_str


    def draw_areagram_data(self):
        """Draw the object data into an XY plot and refresh the plot"""
        self.axes_areagram.clear()
        self.area_bar = self.axes_areagram.pcolormesh(self.area_t, self.area_f / 10, self.ag_matrix, cmap='gray')
        # self.colorbar_area = self.figure_areagram.colorbar(self.area_bar)
        # area_bar = self.axes_areagram.imshow(self.ag_matrix, origin='upper', cmap="gray", aspect='auto')
        # cbar.ax.tick_params(labelsize=7)
        self.axes_areagram.tick_params(axis='both', which='major', labelsize=8)
        self.axes_areagram.set_ylabel('L-G Dist. (norm)', fontname="Arial", fontsize=10, labelpad=10)
        self.axes_areagram.xaxis.set_major_locator(ticker.LinearLocator(6))
        self.canvas_areagram.draw()
        self.canvas_areagram.Refresh()


    def set_areagram_data_plot(self, xvals, area_f, area_t):
        """To set the data coming from main window for plotting """
        self.ag_matrix = xvals
        self.area_f = area_f
        self.area_t = area_t
        self.draw_areagram_data()

    def on_areagram_dclick(self, event):
        """to open new image after double clicking on areagram"""
        if event.dblclick:
            fig = plt.figure()
            fig.add_subplot(111)
            fig.canvas.set_window_title('Areagram: Distance Between G-L')
            plt.pcolormesh(self.area_t, self.area_f / 10, self.ag_matrix, cmap='gray')
            plt.colorbar()
            plt.title('Areagram')
            plt.xlabel('Time')
            plt.ylabel('G-L Dist. (norm)')
            plt.show()

    def on_spectrogram_dclick(self, event):
        """to open new image after double clicking on spectrogram"""
        if event.dblclick:
            fig = plt.figure()
            fig.add_subplot(111)
            fig.canvas.set_window_title('Spectrogram: ')
            data_raw = self.raw_data
            N1 = 1
            N2 = len(data_raw) - 100
            data = np.concatenate((data_raw[N1 - 1:N2], np.zeros(len(data_raw) - N2)), axis=0)
            wsize = 290  # 29 ms window length
            o_lap = np.round(0.9 * wsize)
            self.spec_mat, self.spec_f, self.spec_t, self.spec_img = plt.specgram(x=data, pad_to=512,
                                                                                                    NFFT=wsize,
                                                                                                    window=np.hamming(wsize),
                                                                                                    Fs=10000,
                                                                                                    noverlap=o_lap,
                                                                                                    cmap='gist_yarg')
            plt.colorbar()
            plt.title('Spectrogram')
            plt.xlabel('Time')
            plt.ylabel('Freq')
            plt.show()

