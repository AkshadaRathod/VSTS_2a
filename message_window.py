

import wx
from wx.lib.pubsub import pub


class MessageWindow(wx.Dialog):
    '''Main MessageWindow dialog'''

    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, -1, "Message Window")
        self.SetBackgroundColour("WHITE")
        self.InitUI()

    def InitUI(self):
        Main_Boxsizer = wx.BoxSizer(wx.VERTICAL)
        self.tc = wx.TextCtrl(self, wx.ID_ANY, "", wx.DefaultPosition, size=(-1, 30), style=wx.TE_CENTRE)
        font = wx.Font(12, wx.MODERN, wx.NORMAL, wx.NORMAL)
        self.tc.SetFont(font)
        self.tc.SetBackgroundColour(wx.Colour(236, 236, 231))
        Main_Boxsizer.Add(self.tc, 0, wx.EXPAND | wx.ALL, 15)

        Num_Sizer = wx.BoxSizer(wx.VERTICAL)

        num_grid_sizer = wx.GridSizer(2, 5, 5, 5)

        Num_Sizer.Add(num_grid_sizer, 0, wx.EXPAND | wx.ALL, 5)

        for i in range(0, 10):
            btn = str(i)
            self.num_btn = wx.Button(self, label=btn, size=(45, 35))

            self.num_btn.SetBackgroundColour(wx.Colour(240, 240, 240))
            self.num_btn.Bind(wx.EVT_BUTTON, self.on_button_click)
            num_grid_sizer.Add(self.num_btn, 0, wx.ALIGN_CENTER)

        Main_Boxsizer.Add(Num_Sizer, 0, wx.EXPAND, 1)
        Char_Sizer = wx.BoxSizer(wx.VERTICAL)

        char_grid_sizer = wx.GridSizer(6, 5, 5, 5)

        Num_Sizer.Add(char_grid_sizer, 1, wx.EXPAND | wx.ALL, 5)
        test_list = []
        alpha = '@'
        for i in range(0, 26):
            test_list.append(alpha)
            alpha = str(chr(ord(alpha) + 1))
            self.chr_btn = wx.Button(self, label=alpha, size=(45, 35))
            self.chr_btn.SetBackgroundColour(wx.Colour(236, 236, 231))
            self.chr_btn.Bind(wx.EVT_BUTTON, self.on_button_click)
            char_grid_sizer.Add(self.chr_btn, 0, wx.ALIGN_CENTER)

        Main_Boxsizer.Add(Char_Sizer, 0, wx.EXPAND | wx.ALL, 1)

        # Tool_Sizer = wx.BoxSizer(wx.VERTICAL)
        self.bottomToolbar = wx.ToolBar(self, style=wx.TB_HORIZONTAL | wx.EXPAND | wx.TB_FLAT | wx.TB_NODIVIDER)
        self.bottomToolbar.SetToolBitmapSize(wx.Size(55, 35))
        self.bottomToolbar.SetBackgroundColour("White")

        for i in range(2):
            self.bottomToolbar.AddStretchableSpace()
        img = wx.Image('Lx_Icons/Backspace.png').Scale(80, 50, wx.IMAGE_QUALITY_HIGH)
        b_redo_tool = self.bottomToolbar.AddTool(1, 'Backspace', wx.Bitmap(img), 'Backspace')
        self.Bind(wx.EVT_TOOL, self.on_button_click, b_redo_tool)

        for i in range(2):
            self.bottomToolbar.AddStretchableSpace()

        img = wx.Image('Lx_Icons/Space.png').Scale(80, 50, wx.IMAGE_QUALITY_HIGH)
        b_redo_tool = self.bottomToolbar.AddTool(2, 'Space', wx.Bitmap(img), 'Space')
        self.Bind(wx.EVT_TOOL, self.on_space, b_redo_tool)

        for i in range(2):
            self.bottomToolbar.AddStretchableSpace()

        img = wx.Image('Lx_Icons/Delete.png').Scale(80, 50, wx.IMAGE_QUALITY_HIGH)
        b_redo_tool = self.bottomToolbar.AddTool(3, 'Delete All', wx.Bitmap(img), 'Delete All')
        self.Bind(wx.EVT_TOOL, self.on_clear, b_redo_tool)

        for i in range(2):
            self.bottomToolbar.AddStretchableSpace()

        img = wx.Image('Lx_Icons/Enter.png').Scale(80, 50, wx.IMAGE_QUALITY_HIGH)
        b_redo_tool = self.bottomToolbar.AddTool(4, 'Enter', wx.Bitmap(img), 'Enter')
        self.Bind(wx.EVT_TOOL, self.on_enter, b_redo_tool)
        self.bottomToolbar.Realize()

        Main_Boxsizer.Add(self.bottomToolbar, 0, wx.EXPAND)

        bottomToolbar = wx.ToolBar(self, style=wx.TB_HORIZONTAL | wx.EXPAND | wx.TB_FLAT | wx.TB_NODIVIDER)
        bottomToolbar.SetToolBitmapSize(wx.Size(40, 40))
        bottomToolbar.AddStretchableSpace()
        bottomToolbar.SetBackgroundColour("White")
        img = wx.Image('Lx_Icons/Star.png').Scale(40, 40, wx.IMAGE_QUALITY_HIGH)
        bottomToolbar.AddTool(wx.ID_ANY, 'Star', wx.Bitmap(img))
        img = wx.Image('Lx_Icons/Hindi_A.png').Scale(40, 40, wx.IMAGE_QUALITY_HIGH)
        bottomToolbar.AddTool(wx.ID_ANY, 'Hindi_A', wx.Bitmap(img))
        img = wx.Image('Lx_Icons/A.png').Scale(40, 40, wx.IMAGE_QUALITY_HIGH)
        bottomToolbar.AddTool(wx.ID_ANY, 'A', wx.Bitmap(img))
        img = wx.Image('Lx_Icons/G.png').Scale(40, 40, wx.IMAGE_QUALITY_HIGH)
        bottomToolbar.AddTool(wx.ID_ANY, 'G', wx.Bitmap(img))
        bottomToolbar.AddStretchableSpace()
        bottomToolbar.Realize()

        Main_Boxsizer.Add(bottomToolbar, 1, wx.EXPAND)

        self.SetSizer(Main_Boxsizer)
        self.Layout()
        Main_Boxsizer.Fit(self)

        self.Centre(wx.BOTH)

    def on_button_click(self, event):
        btn = event.GetEventObject()
        label = btn.GetLabel()
        self.current_equation = self.tc.GetValue()
        self.txtVal = self.current_equation + label
        print(self.txtVal)
        self.tc.SetValue(self.txtVal)

    def on_clear(self, event):
        self.tc.Clear()

    def on_space(self, event):
        print("on_space")

    def on_enter(self, event):
        print(self.txtVal)
        msg = self.tc.GetValue()
        pub.sendMessage("SEND_MESSAGE", message=msg)
        # pub.sendMessage("panelListener", message="test2", arg2="2nd argument!")
        self.Close()
