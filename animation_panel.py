#
# import wx
# from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas, FigureCanvasWxAgg
# import matplotlib.image as mpimg
# from matplotlib.figure import Figure
#
# class AnimationPanel(wx.Panel):
#
#     def __init__(self, parent):
#         wx.Panel.__init__(self, parent=parent)
#         self.InitUI()
#
#     def InitUI(self):
#         main_box = wx.BoxSizer(wx.VERTICAL)
#         animation_box = wx.BoxSizer(wx.VERTICAL)
#         self.figure_animation = Figure(figsize=(4,4))
#         self.axes_animation = self.figure_animation.subplots()
#
#         img2 = mpimg.imread('Lx_Icons/Boy.png')
#         self.img = self.axes_animation.imshow(img2)
#         self.canvas_animation = FigureCanvasWxAgg(self, -1, self.figure_animation)
#         # self.axes_animation.get_xaxis().set_visible(False)
#         # self.axes_animation.get_yaxis().set_visible(False)
#         # self.axes_animation.set_axis_off()
#         # self.figure_animation.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
#         #
#         animation_box.Add(self.canvas_animation, 0, wx.EXPAND)
#         main_box.Add(animation_box, 0, wx.EXPAND)
