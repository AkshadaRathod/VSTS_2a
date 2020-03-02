import matplotlib
import wx
from wx.lib.pubsub import pub


class Color_Plate(wx.Dialog):

    def __init__(self, parent,):
        """ Initialize everything here """
        super(Color_Plate, self).__init__(parent)
        self.SetTitle('Color Choice')
        self.InitUI()
        self.Fit()

    def InitUI(self):
        fgSizer = wx.FlexGridSizer(0, 2, 0, 0)
        fgSizer.SetFlexibleDirection(wx.HORIZONTAL)
        fgSizer.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        txt_box = wx.BoxSizer(wx.VERTICAL)

        self.vb = wx.StaticText(self, wx.ID_ANY, u"Vertical Icon Bar", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vb.Wrap(-1)
        self.vb.SetMinSize(wx.Size(-1, 25))

        txt_box.Add(self.vb, 0, wx.ALL, 5)

        self.tb = wx.StaticText(self, wx.ID_ANY, u"Title Bar", wx.DefaultPosition, wx.DefaultSize, 0)
        self.tb.Wrap(-1)
        self.tb.SetMinSize(wx.Size(-1, 25))

        txt_box.Add(self.tb, 0, wx.ALL, 5)

        self.pnl = wx.StaticText(self, wx.ID_ANY, u"Panel", wx.DefaultPosition, wx.DefaultSize, 0)
        self.pnl.Wrap(-1)
        self.pnl.SetMinSize(wx.Size(-1, 25))

        txt_box.Add(self.pnl, 0, wx.ALL, 5)

        self.btnApply = wx.Button(self, wx.ID_ANY, u"Apply", wx.DefaultPosition, wx.DefaultSize, 0)
        self.Bind(wx.EVT_BUTTON, self.on_accept, self.btnApply)
        txt_box.Add(self.btnApply, 0, wx.ALL, 5)


        fgSizer.Add(txt_box, 1, 0, 1)

        pick_box = wx.BoxSizer(wx.VERTICAL)

        self.c_vib = wx.ColourPickerCtrl(self, wx.ID_ANY, wx.BLACK, wx.DefaultPosition, wx.DefaultSize,
                                                   wx.CLRP_DEFAULT_STYLE)
        self.c_vib.SetMinSize(wx.Size(-1, 25))

        pick_box.Add(self.c_vib, 0, wx.ALL, 5)

        self.c_tb = wx.ColourPickerCtrl(self, wx.ID_ANY, wx.BLACK, wx.DefaultPosition, wx.DefaultSize,
                                                   wx.CLRP_DEFAULT_STYLE)
        self.c_tb.SetMinSize(wx.Size(-1, 25))

        pick_box.Add(self.c_tb, 0, wx.ALL, 5)

        self.m_colourPicker8 = wx.ColourPickerCtrl(self, wx.ID_ANY, wx.BLACK, wx.DefaultPosition, wx.DefaultSize,
                                                   wx.CLRP_DEFAULT_STYLE)
        self.m_colourPicker8.SetMinSize(wx.Size(-1, 25))

        pick_box.Add(self.m_colourPicker8, 0, wx.ALL, 5)

        self.btnCancel = wx.Button(self, wx.ID_ANY, u"Cancel", wx.DefaultPosition, wx.DefaultSize, 0)
        pick_box.Add(self.btnCancel, 0, wx.ALL, 5)

        fgSizer.Add(pick_box, 1, 0, 5)

        self.SetSizer(fgSizer)
        self.Layout()
        fgSizer.Fit(self)

        self.Centre(wx.BOTH)

    def on_accept(self, event):
        print("on_accept")
        # colour_rbg = rgb_to_hex(self.c_vib.GetColour())
        colour_vib = self.c_vib.GetColour()
        colour_tb = self.c_tb.GetColour()
        pub.sendMessage("COLOR_CHANGE", value_vib=colour_vib, value_tb=colour_tb)

        self.Destroy()

        # hex_result = "".join([format(val, '02X') for val in colour_rbg])
        # pub.sendMessage("COLOR_CHANGE", value=f"{hex_result}")

        # print(f"#{hex_result}")
        # print(colour_rbg)
