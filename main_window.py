import time
import scipy.io.wavfile as wv
import wx
import numpy as np
from wx.lib.pubsub import pub

from PDFViewer import PDFViewer
from analysis import Analysis_Panel
from analysis_lower_left import Analysis_Lower_Left
from animation__left_panel import AnimationLeftPanel
from animation_right_panel import AnimationRightPanel
from color_plate import Color_Plate
from main_manoj import vsts
from plot_graph import PlotGraph
from plot_panel import PlotPanel
from signal_acquisition import SignalAcquisitionDialog
from user_data import UserData


class main_window(wx.Frame):
    def __init__(self, *args, **kwds):
        kwds["style"] = kwds.get("style" ,0)
        wx.Frame.__init__(self, *args, **kwds)
        self.SetBackgroundColour(wx.Colour("WHITE"))
        self.user_data = [UserData('User 1'), UserData('User 2')]
        self.plot_panel = [None, None]
        self.signal_acqui = [None, None]
        self.analysis_panel = [None, None]
        self.animation_panel = [None, None]

        self.analysis_lower_left = [None, None]
        # Creting the custom title bar
        self.panelTitleBar = wx.Panel(self, wx.ID_ANY)
        self.btnManual = wx.Button(self.panelTitleBar, wx.ID_ANY, "", style=wx.BORDER_NONE | wx.BU_NOTEXT)
        self.btnColor = wx.Button(self.panelTitleBar, wx.ID_ANY, "", style=wx.BORDER_NONE | wx.BU_NOTEXT)
        self.btnMinimize = wx.Button(self.panelTitleBar, wx.ID_ANY, "-", style=wx.BORDER_NONE | wx.BU_NOTEXT)
        self.btnMaximize = wx.Button(self.panelTitleBar, wx.ID_ANY, "[]", style=wx.BORDER_NONE | wx.BU_NOTEXT)
        self.btnExit = wx.Button(self.panelTitleBar, wx.ID_ANY, "", style=wx.BORDER_NONE | wx.BU_NOTEXT)

        self.panelBody = wx.Panel(self, wx.ID_ANY)


        self.Bind(wx.EVT_BUTTON, self.OnBtnExitClick, self.btnExit)
        self.Bind(wx.EVT_BUTTON, self.OnBtnMinimizeClick, self.btnMinimize)
        self.Bind(wx.EVT_BUTTON, self.OnBtnMaximizeClick, self.btnMaximize)
        self.Bind(wx.EVT_BUTTON, self.OnBtnColorClick, self.btnColor)
        self.Bind(wx.EVT_BUTTON, self.OnBtnManualClick, self.btnManual)
        self.panelTitleBar.Bind(wx.EVT_LEFT_DOWN, self.OnTitleBarLeftDown)
        self.panelTitleBar.Bind(wx.EVT_MOTION, self.OnMouseMove)


        self._isClickedDown = False
        self._LastPosition = self.GetPosition()
        self.Maximize(True)

        self.__set_properties()
        self.__do_layout()

        pub.subscribe(self.changeColor, "COLOR_CHANGE")
        pub.subscribe(self.receiveMessage, "SEND_MESSAGE")

    def __set_properties(self):
        self.SetTitle("frame")
        self.btnManual.SetMinSize((22, 22))
        self.btnManual.SetBitmap(wx.Bitmap("Lx_Icons/user-manual.png", wx.BITMAP_TYPE_ANY))
        self.btnColor.SetMinSize((22, 22))
        self.btnColor.SetBitmap(wx.Bitmap("Lx_Icons/Color.png", wx.BITMAP_TYPE_ANY))
        self.btnMinimize.SetMinSize((22, 22))
        self.btnMinimize.SetBitmap(wx.Bitmap("Lx_Icons/Minimize.png", wx.BITMAP_TYPE_ANY))
        self.btnMaximize.SetMinSize((22, 22))
        self.btnMaximize.SetBitmap(wx.Bitmap("Lx_Icons/Maximize.png", wx.BITMAP_TYPE_ANY))
        self.btnExit.SetMinSize((22, 22))
        self.btnExit.SetBitmap(wx.Bitmap("Lx_Icons/Close.png", wx.BITMAP_TYPE_ANY))
        self.panelTitleBar.SetBackgroundColour(wx.Colour(44, 134, 179))
        self.panelBody.SetBackgroundColour(wx.Colour(255, 255, 255))

    def __do_layout(self):

        #Sizers:
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        grid_sizer_1 = wx.FlexGridSizer(2, 1, 0, 0)
        sizerTitleBar = wx.FlexGridSizer(1, 7, 0, 0)

        #Titlebar:
        iconTitleBar = wx.StaticBitmap(self.panelTitleBar, wx.ID_ANY, wx.Bitmap("Lx_Icons/Desktop_Icon22x22.png", wx.BITMAP_TYPE_ANY))
        sizerTitleBar.Add(iconTitleBar, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 1)

        title = wx.StaticText(self.panelTitleBar, wx.ID_ANY, "Visual Speech Training System v2.0 ")
        title.SetForegroundColour(wx.Colour(255, 255, 255))
        title.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizerTitleBar.Add(title, 0, wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM | wx.TOP, 10)
        sizerTitleBar.Add(self.btnManual, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 4)
        sizerTitleBar.Add(self.btnColor, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 4)
        sizerTitleBar.Add(self.btnMinimize, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 4)
        sizerTitleBar.Add(self.btnMaximize, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 4)
        sizerTitleBar.Add(self.btnExit, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 4)
        sizerTitleBar.AddGrowableRow(0)
        sizerTitleBar.AddGrowableCol(1)

        self.panelTitleBar.SetSizer(sizerTitleBar)
        grid_sizer_1.Add(self.panelTitleBar, 1, wx.EXPAND, 0)

        grid_sizer_1.Add(self.panelBody, 1, wx.EXPAND, 0)
        grid_sizer_1.AddGrowableRow(1)
        grid_sizer_1.AddGrowableCol(0)
        main_sizer.Add(grid_sizer_1,0, wx.EXPAND, 0) #minimize


        """Start of frame_box that holds boxsizers which Initialise the left, right toolbars and left, right Panels"""
        frame_box = wx.BoxSizer(wx.VERTICAL)

        top_box = wx.BoxSizer(wx.HORIZONTAL)

        '''Start of left_toolbar_box to hold all the tool on left side of the App'''
        left_toolbar_box = wx.BoxSizer(wx.VERTICAL)

        self.leftToolbar = wx.ToolBar(self, style=wx.TB_VERTICAL | wx.EXPAND | wx.TB_FLAT | wx.TB_NODIVIDER)
        # Add a tool button for clearing the current signal
        for i in range(20):
            self.leftToolbar.AddStretchableSpace()
        # Add a tool button for acquiring a new signal
        img = wx.Image('Lx_Icons/Signal_Normal.png').Scale(100, 60, wx.IMAGE_QUALITY_HIGH)
        l_acquire_tool = self.leftToolbar.AddTool(2, 'Signal', wx.Bitmap(img), 'Acquire new signal')
        self.Bind(wx.EVT_TOOL, self.on_Switch_Left_Panel, l_acquire_tool)
        for i in range(3):
            self.leftToolbar.AddStretchableSpace()

        # Add a tool button for analyzing the current signal
        img = wx.Image('Lx_Icons/Analysis_Normal.png').Scale(100, 60, wx.IMAGE_QUALITY_HIGH)
        l_analyze_tool = self.leftToolbar.AddTool(3, 'Analyze', wx.Bitmap(img), 'Analyze current signal')
        self.Bind(wx.EVT_TOOL, self.on_Switch_Left_Analysis_Panel, l_analyze_tool)
        for i in range(3):
            self.leftToolbar.AddStretchableSpace()

        # Add a tool button for animating the current signal
        img = wx.Image('Lx_Icons/Animation_Disabled.png').Scale(100, 60, wx.IMAGE_QUALITY_HIGH)
        l_animate_tool = self.leftToolbar.AddTool(4, 'Animation', wx.Bitmap(img), 'Animate current signal')
        # self.on_left_animation(None, init=True)
        self.leftToolbar.EnableTool(4, enable=False)
        self.Bind(wx.EVT_TOOL, self.on_Switch_Left_Animation_Panel, l_animate_tool)
        self.leftToolbar.AddStretchableSpace()

        self.leftToolbar.Realize()

        left_toolbar_box.Add(self.leftToolbar, 1, wx.EXPAND, 1)

        self.editLeftMessage = wx.TextCtrl(self, wx.ID_ANY, u"Message", wx.DefaultPosition, wx.Size( 107,40 ), wx.TE_CENTRE|wx.TE_READONLY |wx.TE_RICH2)
        self.editLeftMessage.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        self.editLeftMessage.SetForegroundColour(wx.Colour(255, 255, 255))
        self.editLeftMessage.SetBackgroundColour(wx.Colour(64,134,170,1))
        left_toolbar_box.Add(self.editLeftMessage, 0)

        top_box.Add(left_toolbar_box, 0, wx.ALL | wx.EXPAND, 1)
        '''End of left_toolbar_box'''

        '''Start of left_signal_box to hold PlotPanel and SignalAcquisitionDialog Panel on Left Side'''
        left_signal_box = wx.BoxSizer(wx.VERTICAL)

        self.plot_panel[0] = PlotPanel(self, title=self.user_data[0].user_name)
        left_signal_box.Add(self.plot_panel[0], 1, wx.ALL | wx.EXPAND, 1)  # Expanded

        self.signal_acqui[0] = SignalAcquisitionDialog(self, self.user_data[0])
        self.signal_acqui[0].Hide()
        left_signal_box.Add(self.signal_acqui[0], 1, wx.ALL | wx.EXPAND, 1)

        self.analysis_panel[0] = Analysis_Panel(self, self.user_data[0])
        self.analysis_panel[0].Hide()
        left_signal_box.Add(self.analysis_panel[0], 1, wx.ALL | wx.EXPAND, 1)

        self.analysis_lower_left[0] = Analysis_Lower_Left(self, self.user_data[0])
        self.analysis_lower_left[0].Hide()

        left_signal_box.Add(self.analysis_lower_left[0], 1, wx.ALL | wx.EXPAND, 1)

        self.animation_panel[0] = AnimationLeftPanel(self, self.user_data[0])
        self.animation_panel[0].Hide()
        left_signal_box.Add(self.animation_panel[0], 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER, 1)


        top_box.Add(left_signal_box, 1, wx.ALL | wx.EXPAND, 1)
        '''End of left_signal_box'''

        '''Start of right_signal_box to hold PlotPanel and SignalAcquisitionDialog Panel on Right Side'''
        right_signal_box = wx.BoxSizer(wx.VERTICAL)

        self.plot_panel[1] = PlotPanel(self, title=self.user_data[1].user_name)
        right_signal_box.Add(self.plot_panel[1], 1, wx.ALL | wx.EXPAND, 1)
        self.signal_acqui[1] = SignalAcquisitionDialog(self, self.user_data[1])
        self.signal_acqui[1].Hide()
        right_signal_box.Add(self.signal_acqui[1], 1, wx.ALL | wx.EXPAND, 1)

        self.analysis_panel[1] = Analysis_Panel(self, self.user_data[1])
        self.analysis_panel[1].Hide()
        right_signal_box.Add(self.analysis_panel[1], 1, wx.ALL | wx.EXPAND, 1)

        self.analysis_lower_left[1] = Analysis_Lower_Left(self, self.user_data[1])
        self.analysis_lower_left[1].Hide()
        right_signal_box.Add(self.analysis_lower_left[1], 1, wx.ALL | wx.EXPAND, 1)

        self.animation_panel[1] = AnimationRightPanel(self, self.user_data[0])
        self.animation_panel[1].Hide()
        right_signal_box.Add(self.animation_panel[1], 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER, 1)

        top_box.Add(right_signal_box, 1, wx.ALL | wx.EXPAND, 1)
        '''End of right_signal_box'''


        '''Start of right_toolbar_box to hold all the tool on right side of the App'''
        right_toolbar_box = wx.BoxSizer(wx.VERTICAL)
        self.rightToolbar = wx.ToolBar(self, style=wx.TB_VERTICAL | wx.EXPAND | wx.TB_FLAT | wx.TB_NODIVIDER)

        for i in range(20):
            self.rightToolbar.AddStretchableSpace()


        # Add a tool button for acquiring a new signal
        img = wx.Image('Lx_Icons/Signal_Normal.png').Scale(100, 60, wx.IMAGE_QUALITY_HIGH)
        self.r_acquire_tool = self.rightToolbar.AddTool(101, 'Signal', wx.Bitmap(img), 'Acquire new signal')
        # self.Bind(wx.EVT_TOOL, self.on_acquire_signal_right, self.r_acquire_tool)
        # self.on_right_signal(None, init=True)
        self.Bind(wx.EVT_TOOL, self.on_Switch_Right_Panel, self.r_acquire_tool)
        for i in range(3):
            self.rightToolbar.AddStretchableSpace()
        # self.rightToolbar.AddSeparator()

        # Add a tool button for analyzing the current signal
        img = wx.Image('Lx_Icons/Analysis_Normal.png').Scale(100, 60, wx.IMAGE_QUALITY_HIGH)
        self.r_analyze_tool = self.rightToolbar.AddTool(102, 'Analyze', wx.Bitmap(img), 'Analyze current signal')
        self.Bind(wx.EVT_TOOL, self.on_Switch_Right_Analysis_Panel, self.r_analyze_tool)
        # self.on_right_analysis(None, init=True)
        # self.Bind(wx.EVT_TOOL, self.on_right_analysis, self.r_analyze_tool)
        for i in range(3):
            self.rightToolbar.AddStretchableSpace()
        # self.rightToolbar.AddSeparator()

        # Add a tool button for animating the current signal
        img = wx.Image('Lx_Icons/Animation_Disabled.png').Scale(100, 60, wx.IMAGE_QUALITY_HIGH)
        r_animate_tool = self.rightToolbar.AddTool(103, 'Animation', wx.Bitmap(img), 'Animate current signal')
        self.Bind(wx.EVT_TOOL, self.onClear(), r_animate_tool)
        self.rightToolbar.EnableTool(103, enable=False)
        # self.on_right_animation(None, init=True)
        self.Bind(wx.EVT_TOOL, self.on_Switch_Right_Animation_Panel, r_animate_tool)
        self.rightToolbar.AddStretchableSpace()

        self.rightToolbar.Realize()
        # self.rightToolbar.SetBackgroundColour('#4086aa')  # Setting Background Colour on rightToolbar
        self.leftToolbar.SetBackgroundColour((64,134,170,1))  # Setting Background Colour on leftToolbar
        self.rightToolbar.SetBackgroundColour((64,134,170,1))


        right_toolbar_box.Add(self.rightToolbar, 1, wx.EXPAND, 1)

        self.editRightMessage = wx.TextCtrl(self, wx.ID_ANY, u"Message", wx.DefaultPosition, wx.Size(107, 40),
                                           wx.TE_CENTRE|wx.TE_READONLY |wx.TE_RICH2)
        self.editRightMessage.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        self.editRightMessage.SetForegroundColour(wx.Colour(255, 255, 255))
        self.editRightMessage.SetBackgroundColour(wx.Colour(64, 134, 170, 1))
        right_toolbar_box.Add(self.editRightMessage, 0)


        top_box.Add(right_toolbar_box, 0, wx.ALL | wx.EXPAND, 1)
        '''End of right_toolbar_box'''

        frame_box.Add(top_box, 1, wx.ALL | wx.EXPAND, 1)
        """End of frame_box that holds all boxsizers"""
        main_sizer.Add(frame_box, 1, wx.EXPAND, 0)


        self.SetSizer(main_sizer)
        self.Layout()

    def OnTitleBarLeftDown(self, event):
        self._LastPosition = event.GetPosition()

    def OnBtnExitClick(self, event):
        self.Close()

    def OnBtnMinimizeClick(self, event):
        self.Iconize( True )

    def OnBtnManualClick(self, event):
        import wx.lib.mixins.inspection as WIT
        # app = WIT.InspectableApp(redirect=False)

        pdfV = PDFViewer(None, size=(800, 600))
        pdfV.viewer.UsePrintDirect = False
        pdfV.viewer.LoadFile(r'D:/Akshada_Rathod/VSTS/VSTS Documents/VSTS_2a_Product Specification Document_Draft20dec2019.pdf')
        pdfV.Show()

    def OnBtnColorClick(self, event):
        color_plate = Color_Plate(self)
        color_plate.Centre()
        color_plate.ShowModal()

    def changeColor(self, value_vib, value_tb):
        print("COLOR_CHANGE", value_vib)
        self.rightToolbar.SetBackgroundColour(value_vib)
        self.rightToolbar.Realize()
        self.leftToolbar.SetBackgroundColour(value_vib)
        self.leftToolbar.Realize()
        # self.panelTitleBar.SetBackgroundColour(value_tb)
        # self.SetBackgroundColour(value)

    def OnBtnMaximizeClick(self, event):
        self.Maximize(not self.IsMaximized())

    def OnMouseMove(self, event):
        if event.Dragging():
            mouse_x, mouse_y = wx.GetMousePosition()
            self.Move(mouse_x-self._LastPosition[0],mouse_y-self._LastPosition[1])

    def init_toolbars(self):
        """Initialize the left, right and bottom toolbars"""
        """Start of frame_box that holds boxsizers which Initialise the left, right toolbars and left, right Panels"""
        frame_box = wx.BoxSizer(wx.VERTICAL)

        top_box = wx.BoxSizer(wx.HORIZONTAL)

        '''Start of left_toolbar_box to hold all the tool on left side of the App'''
        left_toolbar_box = wx.BoxSizer(wx.VERTICAL)

        self.leftToolbar = wx.ToolBar(self, style=wx.TB_VERTICAL | wx.EXPAND | wx.TB_FLAT | wx.TB_NODIVIDER)
        # Add a tool button for clearing the current signal
        for i in range(20):
            self.leftToolbar.AddStretchableSpace()

        # Add a tool button for acquiring a new signal
        img = wx.Image('Lx_Icons/Signal_Normal.png').Scale(100, 60, wx.IMAGE_QUALITY_HIGH)
        l_acquire_tool = self.leftToolbar.AddTool(2, 'Signal', wx.Bitmap(img), 'Acquire new signal')
        self.Bind(wx.EVT_TOOL, self.onClear, l_acquire_tool)
        for i in range(3):
            self.leftToolbar.AddStretchableSpace()

        # Add a tool button for analyzing the current signal
        img = wx.Image('Lx_Icons/Analysis_Normal.png').Scale(100, 60, wx.IMAGE_QUALITY_HIGH)
        l_analyze_tool = self.leftToolbar.AddTool(3, 'Analyze', wx.Bitmap(img), 'Analyze current signal')
        self.Bind(wx.EVT_TOOL, self.onClear, l_analyze_tool)
        for i in range(3):
            self.leftToolbar.AddStretchableSpace()

        # Add a tool button for animating the current signal
        img = wx.Image('Lx_Icons/Animation_Disabled.png').Scale(100, 60, wx.IMAGE_QUALITY_HIGH)
        l_animate_tool = self.leftToolbar.AddTool(4, 'Animation', wx.Bitmap(img), 'Animate current signal')
        # self.on_left_animation(None, init=True)
        self.leftToolbar.EnableTool(4, enable=False)
        self.Bind(wx.EVT_TOOL, self.onClear, l_animate_tool)
        self.leftToolbar.AddStretchableSpace()

        self.leftToolbar.Realize()

        left_toolbar_box.Add(self.leftToolbar, 1, wx.EXPAND, 1)
        top_box.Add(left_toolbar_box, 0, wx.ALL | wx.EXPAND, 1)
        '''End of left_toolbar_box'''

        frame_box.Add(top_box, 1, wx.ALL | wx.EXPAND, 1)
        """End of frame_box that holds all boxsizers"""

        self.SetSizer(frame_box)
        self.Layout()

    def onClear(self):
        print("onClear")


    def on_Switch_Left_Panel(self, event):
        """ Switching of panel from plot_panel to signal_acquisition_panel on the left side"""
        if self.plot_panel[0].IsShown() or self.animation_panel[0].IsShown() or self.analysis_lower_left[0].IsShown():
            self.SetTitle("Visual Speech Training System v2.0 : Signal Acquisition")
            self.plot_panel[0].Hide()
            self.analysis_panel[0].Hide()
            self.signal_acqui[0].Show()
            self.animation_panel[0].Hide()
            self.analysis_lower_left[0].Hide()
            img = "Lx_Icons/Signal_Clicked.png"
            png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            self.leftToolbar.SetToolNormalBitmap(id=2, bitmap=png)

            self.enable_left_analysis_normal()
            self.disable_left_animation_normal()
        else:
            self.SetTitle("Visual Speech Training System v2.0")
            self.plot_panel[0].Show()
            # self.left_character.Show()
            self.signal_acqui[0].Hide()

            img = "Lx_Icons/Signal_Normal.png"
            png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            self.leftToolbar.SetToolNormalBitmap(id=2, bitmap=png)

            # self.enable_left_demo_normal()
            self.enable_left_analysis_normal()
            self.enable_left_animation_normal()
        self.Layout()

    def on_Switch_Right_Panel(self, event):
        """ Switching of panel from plot_panel to signal_acquisition_panel on the right side"""
        if self.plot_panel[1].IsShown() or self.animation_panel[1].IsShown() or self.analysis_lower_left[1].IsShown():
            self.SetTitle("Visual Speech Training System v2.0 : Signal Acquisition")
            self.plot_panel[1].Hide()
            self.analysis_panel[1].Hide()
            self.signal_acqui[1].Show()
            self.animation_panel[1].Hide()
            self.analysis_lower_left[1].Hide()
            img = "Lx_Icons/Signal_Clicked.png"
            png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            self.rightToolbar.SetToolNormalBitmap(id=101, bitmap=png)
            # self.disable_right_demo_normal()
            self.enable_right_analysis_normal()
            self.disable_right_animation_normal()
        else:
            self.SetTitle("Visual Speech Training System v2.0")
            self.plot_panel[1].Show()
            self.signal_acqui[1].Hide()
            img = "Lx_Icons/Signal_Normal.png"
            png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            self.rightToolbar.SetToolNormalBitmap(id=101, bitmap=png)
            # self.enable_right_demo_normal()
            self.enable_right_analysis_normal()
            self.enable_right_animation_normal()
        self.Layout()

    ################################################# End of Switching Plot Panel ##############################################################

    def enable_right_demo_normal(self):
        self.rightToolbar.EnableTool(100, enable=True)
        img = "Lx_Icons/Demo_Normal.png"
        png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.rightToolbar.SetToolNormalBitmap(id=100, bitmap=png)

    def enable_right_signal_normal(self):
        self.rightToolbar.EnableTool(101, enable=True)
        img = "Lx_Icons/Signal_Normal.png"
        png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.rightToolbar.SetToolNormalBitmap(id=101, bitmap=png)

    def enable_right_analysis_normal(self):
        self.rightToolbar.EnableTool(102, enable=True)
        img = "Lx_Icons/Analysis_Normal.png"
        png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.rightToolbar.SetToolNormalBitmap(id=102, bitmap=png)

    def enable_right_animation_normal(self):
        self.rightToolbar.EnableTool(103, enable=True)
        img = "Lx_Icons/Animation_Normal.png"
        png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.rightToolbar.SetToolNormalBitmap(id=103, bitmap=png)

    def disable_right_demo_normal(self):
        self.rightToolbar.EnableTool(100, enable=False)
        img = "Lx_Icons/Demo_Disabled.png"
        png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.rightToolbar.SetToolNormalBitmap(id=100, bitmap=png)

    def disable_right_signal_normal(self):
        self.rightToolbar.EnableTool(101, enable=False)
        img = "Lx_Icons/signal_Disabled.png"
        png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.rightToolbar.SetToolNormalBitmap(id=101, bitmap=png)

    def disable_right_analysis_normal(self):
        self.rightToolbar.EnableTool(102, enable=False)
        img = "Lx_Icons/Analysis_Disabled.png"
        png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.rightToolbar.SetToolNormalBitmap(id=102, bitmap=png)

    def disable_right_animation_normal(self):
        self.rightToolbar.EnableTool(103, enable=False)
        img = "Lx_Icons/Animation_Disabled.png"
        png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.rightToolbar.SetToolNormalBitmap(id=103, bitmap=png)

    def enable_left_demo_normal(self):
        self.leftToolbar.EnableTool(1, enable=True)
        img = "Lx_Icons/Demo_Normal.png"
        png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.leftToolbar.SetToolNormalBitmap(id=1, bitmap=png)

    def enable_left_signal_normal(self):
        self.leftToolbar.EnableTool(2, enable=True)
        img = "Lx_Icons/Signal_Normal.png"
        png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.leftToolbar.SetToolNormalBitmap(id=2, bitmap=png)

    def enable_left_analysis_normal(self):
        self.leftToolbar.EnableTool(3, enable=True)
        img = "Lx_Icons/Analysis_Normal.png"
        png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.leftToolbar.SetToolNormalBitmap(id=3, bitmap=png)

    def enable_left_animation_normal(self):
        self.leftToolbar.EnableTool(4, enable=True)
        img = "Lx_Icons/Animation_Normal.png"
        png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.leftToolbar.SetToolNormalBitmap(id=4, bitmap=png)

    def disable_left_demo_normal(self):
        self.leftToolbar.EnableTool(1, enable=False)
        img = "Lx_Icons/Demo_Disabled.png"
        png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.leftToolbar.SetToolNormalBitmap(id=1, bitmap=png)

    def disable_left_signal_normal(self):
        self.leftToolbar.EnableTool(2, enable=False)
        img = "Lx_Icons/signal_Disabled.png"
        png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.leftToolbar.SetToolNormalBitmap(id=2, bitmap=png)

    def disable_left_analysis_normal(self):
        self.leftToolbar.EnableTool(3, enable=False)
        img = "Lx_Icons/Analysis_Disabled.png"
        png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.leftToolbar.SetToolNormalBitmap(id=3, bitmap=png)

    def disable_left_animation_normal(self):
        self.leftToolbar.EnableTool(4, enable=False)
        img = "Lx_Icons/Animation_Disabled.png"
        png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.leftToolbar.SetToolNormalBitmap(id=4, bitmap=png)

    ################################################# Start of Switching Analysis Panel ##############################################################

    def on_Switch_Left_Analysis_Panel(self, event):
        if self.signal_acqui[0].IsShown() or self.animation_panel[0].IsShown():
            self.SetTitle("Visual Speech Training System v2.0 : Analysis")
            self.signal_acqui[0].Hide()
            self.animation_panel[0].Hide()
            self.enable_left_signal_normal()
            self.enable_left_animation_normal()

            f_val = self.user_data[0].get_flag() #we get the value as false

            if not f_val:

                msg = "Please wait while analysis of signal is in progress..."
                # busyDlg = wx.BusyInfo(msg)
                import wx.lib.agw.pybusyinfo as PBI
                busyDlg = PBI.PyBusyInfo(msg, parent=self, title="Processing Files For Analysis")

                uniq_filename = "analysis_process.wav"
                obj = vsts(uniq_filename)
                signal_fs = obj.Fs
                signal_data = obj.filedata #signal main data
                signal_length = obj.filelen #signal number of samples
                time_data, wave_data = obj.wave_detection()  # Check once after dev
                pitch_x_axis, pitch_y_axis = obj.pitch_calculation()
                intensity_x_axis, intensity_y_axis = obj.area_intensity()
                # spectro_matrix, spectro_freq, spectro_time = obj.spectrogram()  # only one matrix is required for now
                ag_matrix, ag_t_spl, ag_f_spl = obj.areagram()
                self.mat_px, self.mat_py, self.K2_POA_pos, self.h2_POA_pos = obj.area_animation()
                data_raw = obj.data_prepare()[0]

                # num_samples = len(wave_data)
                self.start_time = 0
                self.end_time = signal_length / signal_fs

                self.analysis_panel[0].set_data(time_data, wave_data,  self.start_time,  self.end_time)

                self.analysis_panel[0].set_pitch_data(pitch_x_axis, pitch_y_axis,  self.start_time,  self.end_time)
                self.analysis_panel[0].set_intensity_data(intensity_y_axis, intensity_x_axis,  self.start_time,  self.end_time)
                self.analysis_lower_left[0].set_spectrogram_data_plot(data_raw,  self.start_time, self.end_time)
                self.analysis_lower_left[0].set_areagram_data_plot(ag_matrix, ag_f_spl, ag_t_spl)

                s_op2, s_op5, s_op10, s_op20 = obj.time_scale(signal_data)
                wv.write('s_op2.wav', int(signal_fs), np.float32(s_op2))
                wv.write('s_op5.wav', int(signal_fs), np.float32(s_op5))
                wv.write('s_op10.wav', int(signal_fs), np.float32(s_op10))
                wv.write('s_op20.wav', int(signal_fs), np.float32(s_op20))

                del obj

                #new file
                file_name = "s_op2.wav"
                obj_s_op2 = vsts(file_name)
                self.mat_px_s_op2, self.mat_py_s_op2, self.K2_POA_pos_s_op2, self.h2_POA_pos_s_op2 = obj_s_op2.area_animation()
                # self.user_data[0].set_animation_data_s_op2(mat_px, mat_py, K2_POA_pos, h2_POA_pos)

                del busyDlg
                self.analysis_panel[0].Show()

                self.analysis_lower_left[0].Show()

                img = "Lx_Icons/Analysis_Clicked.png"
                png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
                self.leftToolbar.SetToolNormalBitmap(id=3, bitmap=png)
                f_val = True
                self.user_data[0].set_flag(f_val)


            else:
                img = "Lx_Icons/Analysis_Clicked.png"
                png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
                self.leftToolbar.SetToolNormalBitmap(id=3, bitmap=png)
                self.analysis_panel[0].Show()
                self.analysis_lower_left[0].Show()
        else:
            self.SetTitle("Visual Speech Training System v2.0")
            self.plot_panel[0].Show()
            self.analysis_panel[0].Hide()
            self.analysis_lower_left[0].Hide()
            img = "Lx_Icons/Analysis_Normal.png"
            png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            self.leftToolbar.SetToolNormalBitmap(id=3, bitmap=png)
            self.enable_left_signal_normal()
            # self.enable_left_demo_normal()
            self.enable_left_animation_normal()
        self.Layout()


    def on_Switch_Right_Analysis_Panel(self, event):
        if self.signal_acqui[1].IsShown() or self.animation_panel[1].IsShown():
            self.SetTitle("Visual Speech Training System v2.0 : Analysis")
            self.signal_acqui[1].Hide()
            self.animation_panel[1].Hide()
            self.enable_right_signal_normal()
            self.enable_right_animation_normal()

            f_val = self.user_data[1].get_flag()  # we get the value as false

            if not f_val:

                msg = "Please wait while analysis of signal is in progress..."
                # busyDlg = wx.BusyInfo(msg)
                import wx.lib.agw.pybusyinfo as PBI
                busyDlg = PBI.PyBusyInfo(msg, parent=self, title="Processing Files For Analysis")

                uniq_filename = "analysis_process.wav"
                obj = vsts(uniq_filename)
                signal_fs = obj.Fs
                signal_data = obj.filedata  # signal main data
                signal_length = obj.filelen  # signal number of samples
                start = time.time()
                time_data, wave_data = obj.wave_detection()  # Check once after dev
                pitch_x_axis, pitch_y_axis = obj.pitch_calculation()
                intensity_x_axis, intensity_y_axis = obj.area_intensity()
                # spectro_matrix, spectro_freq, spectro_time = obj.spectrogram()  # only one matrix is required for now
                ag_matrix, ag_t_spl, ag_f_spl = obj.areagram()
                self.mat_px, self.mat_py, self.K2_POA_pos, self.h2_POA_pos = obj.area_animation()
                data_raw = obj.data_prepare()[0]
                end = time.time()
                # print(end - start)
                # print(time_data, wave_data, pitch_x_axis, pitch_y_axis, intensity_x_axis, intensity_y_axis, ag_matrix,
                #       ag_t_spl, ag_f_spl)

                # num_samples = len(wave_data)
                self.start_time = 0
                self.end_time = signal_length / signal_fs

                self.analysis_panel[1].set_data(time_data, wave_data, self.start_time, self.end_time)

                self.analysis_panel[1].set_pitch_data(pitch_x_axis, pitch_y_axis, self.start_time, self.end_time)
                self.analysis_panel[1].set_intensity_data(intensity_y_axis, intensity_x_axis, self.start_time,
                                                          self.end_time)
                self.analysis_lower_left[1].set_spectrogram_data_plot(data_raw, self.start_time, self.end_time)
                self.analysis_lower_left[1].set_areagram_data_plot(ag_matrix, ag_f_spl, ag_t_spl)

                s_op2, s_op5, s_op10, s_op20 = obj.time_scale(signal_data)
                wv.write('s_op2.wav', int(signal_fs), np.float32(s_op2))
                wv.write('s_op5.wav', int(signal_fs), np.float32(s_op5))
                wv.write('s_op10.wav', int(signal_fs), np.float32(s_op10))
                wv.write('s_op20.wav', int(signal_fs), np.float32(s_op20))

                del obj

                # new file
                file_name = "s_op2.wav"
                obj_s_op2 = vsts(file_name)
                self.mat_px_s_op2, self.mat_py_s_op2, self.K2_POA_pos_s_op2, self.h2_POA_pos_s_op2 = obj_s_op2.area_animation()
                # self.user_data[0].set_animation_data_s_op2(mat_px, mat_py, K2_POA_pos, h2_POA_pos)

                del busyDlg
                self.analysis_panel[1].Show()
                self.analysis_lower_left[1].Show()
                img = "Lx_Icons/Analysis_Clicked.png"
                png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
                self.leftToolbar.SetToolNormalBitmap(id=102, bitmap=png)
                f_val = True
                self.user_data[1].set_flag(f_val)
            else:
                img = "Lx_Icons/Analysis_Clicked.png"
                png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
                self.rightToolbar.SetToolNormalBitmap(id=102, bitmap=png)
                self.analysis_panel[1].Show()
                self.analysis_lower_left[1].Show()
        else:
            self.SetTitle("Visual Speech Training System v2.0")
            self.plot_panel[1].Show()
            self.analysis_panel[1].Hide()
            self.analysis_lower_left[1].Hide()
            img = "Lx_Icons/Analysis_Normal.png"
            png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            self.rightToolbar.SetToolNormalBitmap(id=102, bitmap=png)
            self.enable_right_signal_normal()
            # self.enable_left_demo_normal()
            self.enable_right_animation_normal()
        self.Layout()

    ################################################# End of Switching Analysis Panel ##############################################################

    ################################################# Start of Switching Aniamtion Panel ##############################################################

    def on_Switch_Left_Animation_Panel(self, event):
        if self.analysis_lower_left[0].IsShown() or self.plot_panel[0].IsShown():
            self.analysis_lower_left[0].Hide()
            self.plot_panel[0].Hide()
            self.analysis_panel[0].Show()
            img = "Lx_Icons/Animation_Clicked.png"
            png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            self.leftToolbar.SetToolNormalBitmap(id=4, bitmap=png)
            self.enable_left_analysis_normal()
            self.animation_panel[0].Show()
            # mat_px, mat_py, K2_POA_pos, h2_POA_pos = self.user_data[0].get_animation_data()
            self.animation_panel[0].set_animation_data_plot(self.mat_px, self.mat_py, self.K2_POA_pos, self.h2_POA_pos, self.end_time )

            # mat_px, mat_py, K2_POA_pos, h2_POA_pos = self.user_data[0].get_animation_data_s_op2()
            self.animation_panel[0].set_animation_data_plot_s_op2(self.mat_px_s_op2, self.mat_py_s_op2, self.K2_POA_pos_s_op2, self.h2_POA_pos_s_op2)
            self.animation_panel[0].init()
        self.Layout()

    def on_Switch_Right_Animation_Panel(self, event):
        if self.analysis_lower_left[1].IsShown() or self.plot_panel[1].IsShown():
            self.analysis_lower_left[1].Hide()
            self.plot_panel[1].Hide()
            self.analysis_panel[1].Show()
            img = "Lx_Icons/Animation_Clicked.png"
            png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            self.rightToolbar.SetToolNormalBitmap(id=103, bitmap=png)
            self.enable_right_analysis_normal()
            self.animation_panel[1].Show()
            self.animation_panel[1].set_animation_data_plot(self.mat_px, self.mat_py, self.K2_POA_pos, self.h2_POA_pos,
                                                            self.end_time)

            self.animation_panel[1].set_animation_data_plot_s_op2(self.mat_px_s_op2, self.mat_py_s_op2,
                                                                  self.K2_POA_pos_s_op2, self.h2_POA_pos_s_op2)
        self.Layout()
    ################################################# End of Switching Aniamtion Panel ##############################################################


    def receiveMessage(self, message):
        # print("receiveMessage", message)

        self.editLeftMessage.SetValue(message)



