import time

import wx
import matplotlib
from matplotlib import animation
from wx.lib.pubsub import pub

matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

class Analysis_Panel(wx.Panel):
    def __init__(self, parent, user_data):
        """ Initialize everything here """
        super(Analysis_Panel, self).__init__(parent,style=wx.BORDER_DOUBLE)
        self.SetBackgroundColour(wx.Colour("White"))
        self.x_vals,self.y_vals = None, None
        # self.speech_x_vals, self.speech_y_vals = None, None
        self.user_data = user_data
        self.figure = Figure(figsize =(1,1))
        # DPI = self.figure.get_dpi()
        # print("DPI:", DPI)
        # DefaultSize = self.figure.get_size_inches()
        # print("Default size in Inches", DefaultSize)
        # print("Which should result in a %i x %i Image" % (DPI * DefaultSize[0], DPI * DefaultSize[1]))
        self.axes, self.axes_intensity, self.axes_pitch = self.figure.subplots(3, sharex='col')
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.figure.subplots_adjust(top=0.98, bottom=0.15) #adjusted correctly
        self.sld = wx.Slider(self, value=0, minValue=0, maxValue=100, size=(545, -1),
                        style=wx.SL_AUTOTICKS | wx.SL_HORIZONTAL | wx.SL_SELRANGE)
        self.sizer.Add(self.sld, 0, wx.LEFT, 77)
        self.sizer.Add(self.canvas, 1, wx.EXPAND )

        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

        # self.InitUI()
        pub.subscribe(self.changePitch, "PITCH_CHANGE")
        pub.subscribe(self.changeLevel, "LEVEL_CHANGE")
        pub.subscribe(self.changeSlider, "SLIDER_CHANGE")
        pub.subscribe(self.playAnimation, "PLAY")
        # pub.subscribe(self.changeLevelRight, "LEVEL_RIGHT_CHANGE")


    def changeSlider(self, value):
        self.sld.SetValue(min(self.sld.GetValue() + 1, 0))
        for i in range(100):
            self.sld.SetValue(min(self.sld.GetValue() + 1, 100))
            time.sleep(0.01)

    def changePitch(self, value):
        self.axes_pitch.set_ylim(0,value)
        self.canvas.draw()
        self.canvas.Refresh()

    def changeLevel(self, value):
        self.axes_intensity.set_ylim(20,value)
        self.canvas.draw()
        self.canvas.Refresh()

    def draw_speech_data(self):
        self.axes.clear()
        self.axes.set_ylim(-1, 1)
        self.axes.set_xlim(self.start_time,self.end_time)
        self.axes.set_ylabel('Sig.(norm)', fontname="Arial", fontsize=10, labelpad=13)
        self.axes.tick_params(axis='both', which='major', labelsize=8)
        self.axes.plot(self.speech_x_vals, self.speech_y_vals, "b", linewidth=0.5)
        self.l, self.v = self.axes.plot(self.speech_x_vals[0], self.speech_y_vals[0], self.speech_x_vals[-1],
                                        self.speech_y_vals[-1], linewidth=0.5, color='red')
        self.axes.grid(True, color='lightgray')
        self.canvas.draw()
        self.canvas.Refresh()


    def set_data(self, xvals, yvals, start, end):
        self.speech_x_vals = xvals
        self.speech_y_vals = yvals
        self.start_time = start
        self.end_time = end
        self.draw_speech_data()


    def get_data(self):
        return self.speech_x_vals, self.speech_y_vals

    def draw_pitch_data(self):
        self.axes_pitch.clear()
        self.axes_pitch.set_ylim(0, 500)
        self.axes_pitch.set_xlim(self.start_time,self.end_time)
        self.axes_pitch.grid(True, color='lightgray')
        self.axes_pitch.set_ylabel('Pitch (norm)', fontname="Arial", fontsize=10, labelpad=9)
        self.axes_pitch.tick_params(axis='both', which='major', labelsize=8)
        self.axes_pitch.set_xlabel('Time (s)', fontname="Arial", fontsize=10)
        # self.axes_pitch.get_xaxis().set_visible(False)
        self.axes_pitch.plot(self.x_vals, self.y_vals, "g", linewidth=0.5)
        self.canvas.draw()
        self.canvas.Refresh()

    def set_pitch_data(self, xvals, yvals, start, end):
        self.x_vals = xvals
        self.y_vals = yvals
        self.start_time = start
        self.end_time = end
        self.draw_pitch_data()

    def draw_intensity_data(self):
        self.axes_intensity.clear()
        self.axes_intensity.set_ylim(20, 80)
        self.axes_intensity.set_xlim(self.start_time,self.end_time)
        self.axes_intensity.grid(True, color='lightgray')
        self.axes_intensity.set_ylabel('Level (norm)', fontname="Arial", fontsize=10, labelpad=15)
        self.axes_intensity.tick_params(axis='both', which='major', labelsize=8)
        self.axes_intensity.plot(self.x_vals, self.y_vals, "b", linewidth=0.5)
        self.axes_intensity.xaxis.set_major_locator(ticker.LinearLocator(6))
        self.canvas.draw()
        self.canvas.Refresh()


    def set_intensity_data(self, xvals, yvals, start, end):
        self.x_vals = xvals
        self.y_vals = yvals
        self.start_time = start
        self.end_time = end
        self.draw_intensity_data()

    def update_line(self, num, line):
        i = self.speech_x_vals[num]
        print("val of i")
        line.set_data([i, i], [self.speech_y_vals[0], self.speech_y_vals[-1]])
        return line

    def playAnimation(self,value):
        self.line_anim = animation.FuncAnimation(fig= self.figure, func = self.update_line, frames=len(self.speech_x_vals),
                                                 init_func=None, fargs=(self.l,), repeat=None)
        print("Animationplayed")













# import wx
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
# import numpy as np
# from signal_acquisition import SignalAcquisitionDialog
# from user_data import UserData
#
#
# class Analysis_Panel(wx.Panel):
#     def __init__(self, parent, user_data):
#         """ Initialize everything here """
#         super(Analysis_Panel, self).__init__(parent,style=wx.BORDER_DOUBLE)
#         self.user_data = [UserData('User 1'), UserData('User 2')]
#         self.x_vals = None
#         self.y_vals = None
#         self.analysis_ui_init()
#
#
#     def analysis_ui_init(self):
#
#         Main_Box = wx.BoxSizer(wx.VERTICAL)
#
#         Top_Box = wx.BoxSizer(wx.VERTICAL)
#
#         signal_box = wx.BoxSizer(wx.VERTICAL)
#
#         self.figure = Figure(figsize=(1,1))
#         self.axes = self.figure.add_subplot(3, 1, 1)
#         self.axes.grid(True, color='lightgray')
#         self.axes.set_ylabel('Sig. (nrm)', fontname="Arial", fontsize=10)
#
#         xvals, yvals = self.user_data[0].get_pitch_data()
#         print("in analysis.py", xvals, yvals)
#         self.axes.plot(xvals, yvals, "b", linewidth=0.5)
#
#
#         # filename = "analysis_process.wav"
#         # obj = vsts(filename)
#         # data, fs = sf.read(filename, dtype='float32')
#         # self.signal_data = data
#         # self.signal_fs = 10000
#         # num_samples = len(data)
#         # self.signal_duration = num_samples / self.signal_fs
#         # self.signal_time = np.linspace(0.0, self.signal_duration, num_samples)
#         # self.axes.plot(self.signal_time, self.signal_data, "b", linewidth=0.5)
#
#
#
#         self.figure.subplots_adjust(hspace=0.5)
#         self.axes_pitch = self.figure.add_subplot(3, 1, 2)
#         self.axes_pitch.grid(True, color='lightgray')
#         self.axes_pitch.set_ylabel('Pitch (Hz)', fontname="Arial", fontsize=10)
#
#         # self.signal_time, self.signal_data = obj.pitch_calculation()
#         # self.axes_pitch.plot(self.signal_time, self.signal_data, "b", linewidth=0.5)
#
#         self.figure.subplots_adjust(hspace=0.5)
#         self.axes_loudness = self.figure.add_subplot(3, 1, 3)
#         self.axes_loudness.grid(True, color='lightgray')
#         self.axes_loudness.set_ylabel('Level (dB)', fontname="Arial", fontsize=10)
#         self.axes_loudness.set_xlabel('Time (s)', fontname="Arial", fontsize=10)
#
#         self.canvas_speech = FigureCanvas(self, -1, self.figure)
#
#
#         signal_box.Add(self.canvas_speech,1, wx.EXPAND | wx.ALL, 1)
#
#         Top_Box.Add(signal_box, 1, wx.EXPAND, 1)
#         Main_Box.Add(Top_Box, 1, wx.EXPAND, 1)
#
#         self.SetSizer(Main_Box)
#         self.Layout()


