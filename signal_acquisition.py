"""
Signal acquisition dialog for VSTS-2 app.
"""

import datetime
import os

import sounddevice as sd
import soundfile as sf
import wx

from analysis import Analysis_Panel
from main_manoj import vsts
from message_window import MessageWindow
from plot_graph import PlotGraph
from user_data import UserData


class SignalAcquisitionDialog(wx.Panel):

    def __init__(self, parent, user_data):
        """ Initialize everything here """
        super(SignalAcquisitionDialog, self).__init__(parent,style=wx.BORDER_DOUBLE)

        # self.SetTitle('Signal acquisition - ' + user_data.user_name)
        self.user_data = user_data
        self.signal_data, self.pitch_data = None, None
        self.xvals, self.yvals = None, None
        self.span = None
        self.data, self.fs = None, None
        self.plot_panel,self.analysis_panel = [None, None],[None, None]

        self.recording_time = 0
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update_progress, self.timer)
        self.SetBackgroundColour(wx.Colour("White"))
        self.InitSignalUI()
        self.Fit()


    def InitSignalUI(self):
        """ Initialize UI elements """

        vert_box = wx.BoxSizer(wx.VERTICAL)

        '''Start of top_tool_box to load, record, start, stop and play the signal'''
        top_tool_box = wx.BoxSizer(wx.VERTICAL)

        '''Start of top_toolbar_box'''
        top_toolbar_box = wx.BoxSizer(wx.VERTICAL)

        '''Start of topToolbar'''
        self.topToolbar = wx.ToolBar(self, style=wx.TB_HORIZONTAL | wx.NO_BORDER | wx.TB_FLAT| wx.TB_NODIVIDER)

        self.topToolbar.SetBackgroundColour('WHITE')
        self.topToolbar.AddStretchableSpace()


        # Add a tool button for Loading the current signal
        img = wx.Image('Lx_Icons/Load_Normal.png').Scale(80,50, wx.IMAGE_QUALITY_HIGH)
        r_play_tool = self.topToolbar.AddTool(wx.ID_ANY, 'Load', wx.Bitmap(img), 'Load Signal')
        self.Bind(wx.EVT_TOOL, self.on_load, r_play_tool)

        # Add a tool button for recording the current signal
        img = wx.Image('Lx_Icons/Record_Normal.png').Scale(80, 50, wx.IMAGE_QUALITY_HIGH)
        r_record_tool = self.topToolbar.AddTool(wx.ID_ANY, 'Record', wx.Bitmap(img),'Record Signal')
        self.Bind(wx.EVT_TOOL, self.on_record_click, r_record_tool)
        for i in range(6):
            self.topToolbar.AddStretchableSpace()


        # Add a tool button for Start Playing the current signal
        img = wx.Image('Lx_Icons/Start_Normal.png').Scale(80, 50, wx.IMAGE_QUALITY_HIGH)
        self.r_start_tool = self.topToolbar.AddTool(100, 'Start', wx.Bitmap(img), 'Start Recording')

        self.Bind(wx.EVT_TOOL, self.on_record_click, self.r_start_tool)

        # Add a tool button for pausing the Recording
        img = wx.Image('Lx_Icons/Stop_Normal.png').Scale(80, 50, wx.IMAGE_QUALITY_HIGH)
        r_stop_tool = self.topToolbar.AddTool(101, 'Stop', wx.Bitmap(img), 'Stop Recording')
        self.Bind(wx.EVT_TOOL, self.on_accept, r_stop_tool)

        # Add a tool button for Playing sound of the signal plotted on Upper Graph
        img = wx.Image('Lx_Icons/Play_Normal_Top.png').Scale(80, 50, wx.IMAGE_QUALITY_HIGH)
        r_play_tool = self.topToolbar.AddTool(102, 'Play', wx.Bitmap(img), 'Play Recording')
        self.Bind(wx.EVT_TOOL, self.on_top_play_click, r_play_tool)
        for i in range(3):
            self.topToolbar.EnableTool(i + 100, False)
        self.topToolbar.Realize()

        top_toolbar_box.Add(self.topToolbar, 0, wx.EXPAND | wx.LEFT, 45 )
        '''End of topToolbar'''

        top_tool_box.Add(top_toolbar_box, 1, wx.EXPAND | wx.TOP , 15)
        '''End of top_toolbar_box'''

        vert_box.Add(top_tool_box, 0, wx.EXPAND| wx.BOTTOM , 10)
        '''End of top_tool_box'''

        '''Start of middle_tool_box to initialize progressbar, and plotgraph'''
        middle_tool_box = wx.BoxSizer(wx.VERTICAL)

        self.record_box = wx.BoxSizer(wx.VERTICAL)
        self.progress_bar = wx.Gauge(self, range=self.user_data.signal_duration, style=wx.GA_HORIZONTAL|wx.GA_SMOOTH)
        self.record_box.Add(self.progress_bar, 0, wx.EXPAND |wx.LEFT|wx.RIGHT, 70)
        self.record_box.AddSpacer(2)
        # self.bar = wx.Image("Lx_Icons/Line_Bar.png", wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        # self.Line_bar = wx.StaticBitmap(self, -1, self.bar)
        # self.record_box.Add(self.Line_bar,0, wx.EXPAND |wx.LEFT|wx.RIGHT, 70)

        middle_tool_box.Add(self.record_box, 0, wx.EXPAND)


        graph_box = wx.BoxSizer(wx.VERTICAL)
        self.plot_graph = PlotGraph(self)
        graph_box.Add(self.plot_graph, 0, wx.EXPAND)

        middle_tool_box.Add(graph_box, 0, wx.EXPAND| wx.TOP, 15)

        vert_box.Add(middle_tool_box, 0, wx.EXPAND)
        '''End of middle_tool_box'''

        '''Start of bottom_tool_box to select, play, reset, save and proceed the signal'''
        bottom_tool_box = wx.BoxSizer(wx.VERTICAL)

        bottom_toolbar_box = wx.BoxSizer(wx.VERTICAL)

        self.bottomToolbar = wx.ToolBar(self, style=wx.TB_HORIZONTAL | wx.NO_BORDER | wx.TB_FLAT | wx.TB_NODIVIDER)
        self.bottomToolbar.SetBackgroundColour('WHITE')
        self.bottomToolbar.AddStretchableSpace()

        # Add a tool button for selecting region from upper graph
        img = wx.Image('Lx_Icons/Select_Normal.png').Scale(80, 50, wx.IMAGE_QUALITY_HIGH)
        r_play_tool = self.bottomToolbar.AddTool(103, 'Select', wx.Bitmap(img),'Select signal')
        self.Bind(wx.EVT_TOOL, self.on_select, r_play_tool)

        self.bottomToolbar.AddStretchableSpace()

        # Add a tool button for Playing the sound signal from lower graph
        img = wx.Image('Lx_Icons/Play_Normal_Bottom.png').Scale(80, 50, wx.IMAGE_QUALITY_HIGH)
        r_play_tool = self.bottomToolbar.AddTool(104, 'Play', wx.Bitmap(img), 'Play signal')
        self.Bind(wx.EVT_TOOL, self.on_bottom_play_click, r_play_tool)

        self.bottomToolbar.AddStretchableSpace()

        # Add a tool button for resetting the signal from lower graph
        img = wx.Image('Lx_Icons/Reset_Normal.png').Scale(80, 50, wx.IMAGE_QUALITY_HIGH)
        r_play_tool = self.bottomToolbar.AddTool(105, 'Reset', wx.Bitmap(img), 'Reset signal')
        self.Bind(wx.EVT_TOOL, self.on_reset_click, r_play_tool)

        self.bottomToolbar.AddStretchableSpace()

        # Add a tool button for Playing the current signal
        img = wx.Image('Lx_Icons/Save_Normal.png').Scale(80, 50, wx.IMAGE_QUALITY_HIGH)
        r_play_tool = self.bottomToolbar.AddTool(106, 'Save', wx.Bitmap(img),'Save signal')
        self.Bind(wx.EVT_TOOL, self.onSaveButton, r_play_tool)

        self.bottomToolbar.AddStretchableSpace()

        # Add a tool button for saving the current signal
        # img = wx.Image('Lx_Icons/Proceed_Normal.png').Scale(80, 50, wx.IMAGE_QUALITY_HIGH)
        # r_proceed_tool = self.bottomToolbar.AddTool(107, 'Proceed', wx.Bitmap(img),'Proceed signal')
        # self.Bind(wx.EVT_TOOL, self.on_proceed, r_proceed_tool)
        for i in range(4):
            self.bottomToolbar.EnableTool(i + 103, False)

        self.bottomToolbar.Realize()

        bottom_toolbar_box.Add(self.bottomToolbar, 0, wx.EXPAND)

        bottom_tool_box.Add(bottom_toolbar_box, 1, wx.EXPAND | wx.TOP |wx.BOTTOM, 20)
        save_box = wx.BoxSizer(wx.HORIZONTAL)
        font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)



        self.FileNameText = wx.StaticText(self, -1, "File Name:\t", style=wx.ALIGN_CENTER_VERTICAL)
        self.FileNameText.SetFont(font)

        save_box.Add(self.FileNameText,0, wx.EXPAND, 1)

        left_box = wx.BoxSizer(wx.VERTICAL)

        self.tcSpeaker = wx.TextCtrl(self, wx.ID_ANY, u"Name", wx.DefaultPosition, wx.Size( -1,18 ), wx.TE_CENTRE)
        self.tcSpeaker.SetFont(font)

        left_box.Add(self.tcSpeaker, 0, wx.ALL, 1)

        self.txtSpeaker = wx.StaticText(self, wx.ID_ANY, u"Speaker", wx.DefaultPosition, wx.DefaultSize,
                                        wx.ALIGN_CENTRE)
        self.txtSpeaker.Wrap(-1)
        self.txtSpeaker.SetFont(font)
        left_box.Add(self.txtSpeaker, 0, wx.LEFT, 40)

        save_box.Add(left_box, 0, wx.EXPAND, 1)

        self.b = wx.StaticText(self, -1, "_", style=wx.ALIGN_CENTER_VERTICAL)
        save_box.Add(self.b, 0, wx.EXPAND, 1)

        mid_box = wx.BoxSizer(wx.VERTICAL)
        date_val = str(datetime.datetime.now().date()).replace('-', '_')
        self.tcDate = wx.TextCtrl(self, wx.ID_ANY, date_val, wx.DefaultPosition, wx.Size( -1,18 ), wx.TE_CENTRE)
        self.tcDate.SetFont(font)

        mid_box.Add(self.tcDate, 0, wx.ALL, 1)

        self.txtDate = wx.StaticText(self, wx.ID_ANY, u"Date", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTRE)
        self.txtDate.Wrap(-1)
        self.txtDate.SetFont(font)
        mid_box.Add(self.txtDate, 0, wx.LEFT, 40)

        save_box.Add(mid_box, 0, wx.EXPAND, 1)
        self.b1 = wx.StaticText(self, -1, "_", style=wx.ALIGN_CENTER_VERTICAL)
        save_box.Add(self.b1, 0, wx.EXPAND, 1)
        right_box = wx.BoxSizer(wx.VERTICAL)
        time_val = str(datetime.datetime.now().strftime('%H_%M_%S'))
        self.tcTime = wx.TextCtrl(self, wx.ID_ANY, time_val, wx.DefaultPosition,  wx.Size( -1,18 ), wx.TE_CENTRE,)
        self.tcTime.SetFont(font)

        right_box.Add(self.tcTime, 0, wx.ALL, 1)

        self.txtTime = wx.StaticText(self, wx.ID_ANY, u"Time", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTRE)
        self.txtTime.Wrap(-1)
        self.txtTime.SetFont(font)

        right_box.Add(self.txtTime, 0, wx.LEFT, 40)

        save_box.Add(right_box, 0, wx.EXPAND, 1)

        self.save_file_box = wx.BoxSizer(wx.VERTICAL)
        self.save_sound_file = wx.StaticText(self, -1, " ")
        self.save_file_box.Add(self.save_sound_file)
        # save_box.Add(self.save_file_box, 0, wx.EXPAND, 1)
        bottom_tool_box.Add(save_box, 0, wx.EXPAND | wx.LEFT, 55)
        bottom_tool_box.AddSpacer(10)
        bottom_tool_box.Add(self.save_file_box, 0, wx.EXPAND | wx.LEFT, 55)

        vert_box.Add(bottom_tool_box, 0, wx.EXPAND)
        '''End of bottom_tool_box'''

        self.SetSizer(vert_box)
        self.Refresh()

    def onSaveButton(self, event):
        print("onsave")
        new_path = "C:/VSTS_Recorded_Sound"

        try:
            if not os.path.exists(new_path):
                os.makedirs(new_path)
        except OSError:
            print("Creation of the directory %s failed" % new_path)
        else:
            print("Successfully created the directory %s " % new_path)

        uniq_filename = str(self.tcSpeaker.GetValue()) + '_' + str(self.tcDate.GetValue()) + '_' + str(self.tcTime.GetValue()) + '.wav'
        completeName = os.path.join(new_path, uniq_filename)

        file1 = open(completeName, "wb")

        self.data_bottom = self.plot_graph.get_audio_data()
        self.fs_bottom = self.user_data.signal_fs
        sf.write(file1, self.data_bottom, self.fs_bottom)
        font1 = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)


        # font = wx.Font(10, wx.ROMAN, wx.ITALIC, wx.NORMAL)
        self.save_sound_file.SetFont(font1)
        self.save_sound_file.SetLabel("File Save as: C:/VSTS_Recorded_Sound/"+uniq_filename)
        # self.save_sound_file = wx.StaticText(self, -1, "File Save as: C:/VSTS_Recorded_Sound/"+uniq_filename)

        # self.save_file_box.Add(self.save_sound_file, 0, wx.TOP, 100)

    def on_message_window(self, e):
        message_dialog = MessageWindow(self)
        message_dialog.Centre()
        message_dialog.ShowModal()
        message_dialog.Destroy()

    def update_progress(self, e):
        self.recording_time += 1
        self.progress_bar.SetValue(self.recording_time)

        if self.recording_time == 10:  # 11-11-2019 by AR
            self.timer.Stop()

        # self.progress_bar.SetValue(0)
        # self.recording_time += 1
        # for i in range(0, int(self.user_data.signal_duration + 1)):
        #     self.recording_time = int(self.user_data.signal_duration + 1)
        #     self.progress_bar.SetValue(self.recording_time)
        #     self.timer.Stop()


    def on_record_click(self, e):
        for i in range(3):
            self.topToolbar.EnableTool(i+100, True)
        for i in range(5):
            self.bottomToolbar.EnableTool(i + 103, True)
        if self.signal_data is None: # Record button
            self.timer.Start(1000)
            duration = self.user_data.signal_duration
            self.fs = self.user_data.signal_fs
            num_samples = (duration*self.fs+1)
            self.signal_data = sd.rec(int(num_samples), samplerate=self.fs, channels=1)

    def on_accept(self, e):
        """ Save changes made before closing the dialog. Ignore event if
         signal being recorded. """
        print("stop button clicked")
        if not self.timer.IsRunning():
            self.user_data.set_signal_data(self.signal_data)
            xvals, yvals = self.user_data.get_signal_data()
            self.plot_graph.set_data(xvals, yvals)
        data_set, fs_set = self.set_audio_data_fs(self.signal_data, self.fs)

            # if self.signal_data is not None:
            #     self.user_data.set_signal_data(self.signal_data)
            #     xvals, yvals = self.user_data.get_signal_data()
            #     self.plot_graph.set_data(xvals, yvals)
            # data_set, fs_set = self.set_audio_data_fs(self.signal_data, self.fs)
            # self.Destroy()


    def on_cancel(self, e):
        """ Ignore changes before closing the dialog. Ignore event if
         signal is being recorded. """
        if not self.timer.IsRunning():
            if self.signal_data is not None:
                dlg = wx.MessageDialog(self, 'Do you want to discard unsaved modifications?',
                                       'Close?', style=wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING)
                # If the user chooses not to exit application, veto the event and return without doing anything
                if dlg.ShowModal() != wx.ID_YES:
                    return

            self.Destroy()

    def on_load(self, e):
        """load wav files from the directory plays the audio and plots the graph"""
        dialog = wx.FileDialog(None, 'Load Audio Clip', wildcard="WAV files (*.wav)|*.wav",style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            filename = dialog.GetPath()
            data, fs = sf.read(filename, dtype='float32')
            data_set, fs_set = self.set_audio_data_fs(data, fs)
            sd.play(data_set, fs_set)
            self.user_data.set_signal_data(data)
            t, y = self.user_data.get_signal_data()
            self.plot_graph.set_data(t, y)
        dialog.Destroy()

        for i in range(5):
            self.bottomToolbar.EnableTool(i + 103, True)
        self.topToolbar.EnableTool(102, True)

    def on_open(self, e):
        print("on_open")
        """Open wav file and plots the graph"""

    # def on_proceed(self, e):
    #     self.data_bottom = self.plot_graph.get_audio_data()
    #     self.fs_bottom = self.user_data.signal_fs
    #
    #     uniq_filename = 'analysis_process.wav'
    #     sf.write(uniq_filename, self.data_bottom, self.fs_bottom)
    #     obj = vsts(uniq_filename)
    #
    #     time_data, wave_data = obj.wave_detection()  # Check once after dev
    #     pitch_x_axis, pitch_y_axis = obj.pitch_calculation()
    #     intensity_x_axis, intensity_y_axis = obj.area_intensity()
    #     self.user_data.set_wave_pitch_intensity_data(time_data, wave_data, pitch_x_axis, pitch_y_axis, intensity_x_axis, intensity_y_axis)
    #
    #     spectro_matrix, spectro_freq, spectro_time = obj.spectrogram()  # only one matrix is required for now
    #     ag_matrix, ag_t_spl, ag_x_spl = obj.areagram()
    #     self.user_data.set_area_spectro_area_data(spectro_matrix, spectro_freq, spectro_time, ag_matrix, ag_t_spl, ag_x_spl )
    #
    #     mat_px, mat_py, K2_POA_pos, h2_POA_pos = obj.area_animation()
    #     self.user_data.set_animation_data(mat_px, mat_py, K2_POA_pos, h2_POA_pos)
    #
    #     del obj
    #
    #     #Normalization of analysis graph x axis is done here
    #     time = self.plot_graph.start_end_time()
    #     start = 0
    #     end = time
    #     self.user_data.set_axis_start_end_time(start,end)

    def on_top_play_click(self, e):
        """set the daya to play upper graph audio"""
        top_x, top_y = self.set_audio_data_fs(self.data, self.fs)
        sd.play(top_x, top_y)

    def on_bottom_play_click(self, e):
        """get and set the data to play lower graph audio"""
        self.data_bottom = self.plot_graph.get_audio_data()
        # self.user_data.set_audio_datafs(self.data_bottom)
        self.fs_bottom = self.user_data.signal_fs
        sd.play(self.data_bottom, self.fs_bottom)

    def on_reset_click(self, e):
        print('About on_reset_click')
        self.plot_graph.clear_pitch_graph_data()

    def on_select(self, e):
        self.plot_graph.span_selection_init()
        self.flag_val = False
        self.user_data.set_flag(self.flag_val)

    def on_clear_click(self, e):
        self.plot_graph.figure.clear()
        print('About on_clear_click')

    def get_audio_data_fs(self): #avoid confusion so TODO check once
        print("getter method called")
        return self.data, self.fs

    def set_audio_data_fs(self, d, s):#avoid confusion so TODO check once
        """setting values of upper graph"""
        self.data = d
        self.fs = s
        return d, s

