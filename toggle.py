import wx


class Toggle(object):
    def __init__(self, tool_name):
        self.tool_name = tool_name
        print("Toggle")

    def ToggleTest(self, init=False):
        if init:
            self.Test = False
        else:
            self.Test = not self.Test
        print("Demo status = ", self.Test)
        if self.Test:
            img = "Icons/Demo_Disabled.png"
            png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            self.tool_name.SetToolNormalBitmap(id=2001, bitmap=png)
            self.tool_name.EnableTool(2001, False)

        else:
            img = "Icons/Demo_Normal.png"
            png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            self.tool_name.SetToolNormalBitmap(id=2001, bitmap=png)

