
import wx
import matplotlib

from user_data import UserData

matplotlib.use('WXAgg')

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import soundfile as sf
# from matplotlib.widgets import SpanSelector  # 13-11-2019 by AR
import matplotlib.widgets as mwidgets

class PlotGraph(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        super(PlotGraph, self).__init__(parent, *args, **kwargs)

        self.x_vals = np.arange(0, 10, 0.01)
        self.y_vals = np.zeros(len(self.x_vals))

        self.InitUI()
        self.span_select = None
        self.x_data = None
        self.y_fs = None
        self.coords_x = []
        self.coords_y = []
        self.line1, = self.axes.plot(self.x_vals, self.y_vals, "b", linewidth=0.5)
        self.line2, = self.axes_pitch.plot(self.x_vals, self.y_vals, "g", linewidth=0.5)

    def InitUI(self):
        main_box = wx.BoxSizer(wx.VERTICAL)

        speech_box = wx.BoxSizer(wx.VERTICAL)
        self.figure = Figure(figsize=(2, 2))
        # self.figure.subplots_adjust(right=0.95, wspace=None, hspace=None)
        self.axes = self.figure.add_subplot(111)
        self.canvas_speech = FigureCanvas(self, -1, self.figure)
        self.axes.set_xlim(0, 10)
        self.axes.set_ylim(-1, 1)
        self.axes.grid(True, color='lightgray')
        self.axes.set_ylabel('Sig.(norm)', fontname="Arial", fontsize=10)
        self.axes.set_xlabel('Time(s)', fontname="Arial", fontsize=10)
        speech_box.Add(self.canvas_speech, 0, wx.EXPAND)
        main_box.Add(speech_box, 0, wx.EXPAND, 1)

        # time_status_box = wx.BoxSizer(wx.HORIZONTAL)
        #
        # left_text_box = wx.BoxSizer(wx.HORIZONTAL)
        #
        # self.txtstart = wx.StaticText(self, wx.ID_ANY, u"  Start Time:", pos=(0,0), size=(200,20), style=wx.ALIGN_CENTER )
        # self.font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        # self.txtstart.SetFont(self.font)
        # self.txtstart.Wrap(-1)
        # left_text_box.Add(self.txtstart, 0, wx.ALL, 10)
        #
        # self.hiddenStartTime = wx.StaticText(self, -1, label = "")
        # left_text_box.Add(self.hiddenStartTime)
        #
        # time_status_box.Add(left_text_box, 1, wx.EXPAND, 1)
        #
        # middle_text_box = wx.BoxSizer(wx.HORIZONTAL)
        #
        # self.txtEnd = wx.StaticText(self, wx.ID_ANY, u"  End Time:", pos=(0,0), size=(200,20), style=wx.ALIGN_CENTER)
        #
        # self.txtEnd.SetFont(self.font)
        # self.txtEnd.Wrap(-1)
        # middle_text_box.Add(self.txtEnd, 0, wx.ALL, 10)
        #
        # time_status_box.Add(middle_text_box, 1, wx.EXPAND, 1)
        #
        # right_text_box = wx.BoxSizer(wx.HORIZONTAL)
        #
        # self.txtDuration = wx.StaticText(self, wx.ID_ANY, u"  Selected Duration:", pos=(0,0), size=(200,20), style=wx.ALIGN_LEFT )
        # self.txtDuration.SetFont(self.font)
        # self.txtDuration.Wrap(-1)
        # right_text_box.Add(self.txtDuration, 0, wx.ALL, 10)

        # time_status_box.Add(right_text_box, 1, wx.EXPAND,1)

        # main_box.Add(time_status_box, 1, wx.EXPAND| wx.TOP,15)

        pitch_box = wx.BoxSizer(wx.VERTICAL)
        self.figure = Figure(figsize=(2, 2))
        self.axes_pitch = self.figure.add_subplot(111)
        # self.figure.subplots_adjust(right=0.90, wspace=None, hspace=None)
        self.canvas_pitch = FigureCanvas(self, -1, self.figure)
        self.axes_pitch.set_xlim(0, 10)
        self.axes_pitch.set_ylim(-1.0, 1.0)
        self.axes_pitch.grid(True, color='lightgray')
        self.axes_pitch.set_ylabel('Selected Sig.(norm)', fontname="Arial", fontsize=10)
        self.axes_pitch.set_xlabel('Time(s)', fontname="Arial", fontsize=10)
        pitch_box.Add(self.canvas_pitch, 0, wx.EXPAND)
        main_box.Add(pitch_box, 0, wx.EXPAND, 1)

        self.SetSizer(main_box)


    def draw_speech_data(self):
        self.axes.clear()
        self.axes.grid(True, color='lightgray')
        self.axes.set_ylabel('Sig.(norm)', fontname="Arial", fontsize=10)
        self.axes.set_xlabel('Time(s)', fontname="Arial", fontsize=10)
        self.axes.set_ylim(-1, 1)
        self.axes.set_xlim(0, 10)
        self.axes.plot(self.x_vals, self.y_vals, "b", linewidth=0.5)
        self.canvas_speech.draw()
        self.canvas_speech.Refresh()

    def set_data(self, xvals, yvals):
        self.x_vals = xvals
        self.y_vals = yvals
        self.draw_speech_data()

    def draw_pitch_data(self):
        self.axes_pitch.clear()
        self.axes_pitch.plot(self.x_vals, self.y_vals, "g", linewidth=0.5)
        self.canvas_pitch.draw()
        self.canvas_pitch.Refresh()

    def set_pitch_data(self, xvals, yvals):
        self.x_vals = xvals
        self.y_vals = yvals
        self.draw_pitch_data()


    def on_select_span(self, min_val, max_val):
        self.indmin, self.indmax = np.searchsorted(self.x_vals, (min_val, max_val))
        self.indmax = min(len(self.x_vals) - 1, self.indmax)
        self.thisx = self.x_vals[self.indmin:self.indmax]
        self.thisy = self.y_vals[self.indmin:self.indmax]
        self.coords_x = self.thisx
        self.coords_y = self.thisy
        self.line2.set_data(self.thisx, self.thisy)
        self.fs_bottom = 10000
        uniq_filename = 'analysis_process.wav'
        sf.write(uniq_filename, self.thisy, self.fs_bottom)

        self.axes_pitch.set_xlim(self.thisx[0], self.thisx[-1]) # thisx[0]= Start Time and thisx[-1] = End Time

        self.axes_pitch.set_ylim(self.thisy.min(), self.thisy.max())
        self.figure.canvas.draw_idle()
        return min_val, max_val

    def get_audio_data(self):
        return self.coords_y

    def start_end_time(self):
        return self.thisx[-1]-self.thisx[0]

    def get_audio_data_xy(self):
        return self.coords_x, self.coords_y

    def clear_pitch_graph_data(self):
        print("clear")
        # self.axes_pitch.clear()
        # self.canvas_pitch.draw()
        # self.axes.grid(True, color='lightgray')
        # self.axes.set_ylabel('Sig.(norm)', fontname="Arial", fontsize=10)
        # self.axes.set_xlabel('Time(s)', fontname="Arial", fontsize=10)
        # self.axes.set_ylim(-1, 1)
        # self.axes.set_xlim(0, 10)
        # self.axes_pitch.plot(self.x_vals, self.y_vals, "g", linewidth=0.5)
        # self.canvas_pitch.Refresh()

    def span_selection_init(self):
        self.span = mwidgets.SpanSelector(self.axes, self.on_select_span, 'horizontal', useblit=True,
                            rectprops=dict(alpha=0.5, facecolor='red'))
        # self.span_select = mwidgets.SpanSelector(self.axes, self.on_select_span, 'horizontal', useblit=True,
        #                                 rectprops=dict(facecolor='red', edgecolor='black', alpha=0.5, fill=True))

