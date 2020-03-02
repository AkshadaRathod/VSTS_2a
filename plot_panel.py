# -*- coding: utf-8 -*-
"""
@author: AnuragSharma

This file encapsulates a Matplotlib panel into a class. This class can be
instantiated and added as a WxPython widget. The widget displays a 2D XY plot.
The values of the plot can be changed after instantiation.
"""

import wx
import matplotlib

from animation__left_panel import AnimationLeftPanel

matplotlib.use('WXAgg')

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class PlotPanel(wx.Panel):
    """
    This class defines a 2D XY plot widget based on a Wx Panel widget.
    """

    def __init__(self, parent, title=None):
        """
        Initialize various plotting widgets and data
        """
        super(PlotPanel, self).__init__(parent,style=wx.BORDER_DOUBLE)

        self.title = title
        self.figure = Figure(figsize=(1,1))
        # self.axes = self.figure.add_subplot(3, 1, 1)
        # self.figure.subplots_adjust(hspace=0.5)  # 16-11-2019 by AR spacing
        # self.axes_pitch = self.figure.add_subplot(3, 1, 2)  # 16-11-2019 by AR
        # self.figure.subplots_adjust(hspace=0.5)  # 19-11-2019 by AR spacing
        # self.axes_loudness = self.figure.add_subplot(3, 1, 3)  # 19-11-2019 by AR
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 2)
        self.SetSizer(self.sizer)

        self.x_vals = np.arange(0, 5, 0.01)
        self.y_vals = np.zeros(len(self.x_vals))
        # self.draw_pitch()
        # self.draw()
        # self.draw_loudness()

    def draw_pitch(self):
        """
        Draw the object data into an XY plot and refresh the plot
        """
        self.axes_pitch.clear()
        self.axes_pitch.grid(True, color='lightgray')
        self.axes_pitch.set_ylabel('Pitch (Hz)')
        self.axes_pitch.set_ylim(100, 500)
        self.axes_pitch.plot(self.x_vals, self.y_vals, "b", linewidth=0.5)
        self.canvas.draw()
        self.canvas.Refresh()

    def draw_loudness(self):
        """
        Draw the object data into an XY plot and refresh the plot
        """
        self.axes_loudness.clear()
        self.axes_loudness.grid(True, color='lightgray')
        self.axes_loudness.set_ylabel('Level (dB)')
        self.axes_loudness.set_xlabel('Time (s)')
        self.axes_loudness.set_ylim(0, 100)
        self.axes_loudness.plot(self.x_vals, self.y_vals, "b", linewidth=0.5)
        self.canvas.draw()
        self.canvas.Refresh()

    def draw(self):
        """
        Draw the object data into an XY plot and refresh the plot
        """
        self.axes.clear()
        # self.axes.set_title(self.title)
        self.axes.grid(True, color='lightgray')
        self.axes.set_ylabel('Amplitude')
        self.axes.set_ylim(-1, 1)
        self.axes.plot(self.x_vals, self.y_vals, "b", linewidth=0.5)
        self.canvas.draw()
        self.canvas.Refresh()

    def set_data(self, xvals, yvals):  # update
        """
        Set new data for the object and redraw the XY plot
        """
        self.x_vals = xvals
        self.y_vals = yvals
        self.draw()

    def set_pitch_data(self, xvals, yvals):  # update
        """
        Set new data for the object and redraw the XY plot
        """
        self.x_vals = xvals
        self.y_vals = yvals
        self.draw_pitch()

    def set_level_data(self, xvals, yvals):  # update
        """
        Set new data for the object and redraw the XY plot
        """
        self.x_vals = xvals
        self.y_vals = yvals
        self.draw_loudness()

