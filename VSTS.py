import wx
from main_window import main_window


def main():
    """ Initiate the app and the UI """
    app = wx.App()
    ex = main_window(None)
    ex.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
