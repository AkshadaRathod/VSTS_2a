import numpy as np
import wx
from matplotlib import animation
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas, FigureCanvasWxAgg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.figure import Figure
import sounddevice as sd
from wx.lib.pubsub import pub
import wx.lib.agw.pygauge as PG
from PIL import Image


class AnimationRightPanel(wx.Panel):

    def __init__(self, parent, user_data):
        """ Initialize everything here """
        super(AnimationRightPanel, self).__init__(parent, style=wx.BORDER_DOUBLE)
        self.SetBackgroundColour(wx.Colour("White"))

        self.user_date = user_data
        self.InitSignalUI()
        self.Fit()

    def InitSignalUI(self):
        vert_box = wx.BoxSizer(wx.VERTICAL)

        '''Start of middle_tool_box to initialize plotanimation'''
        middle_tool_box = wx.BoxSizer(wx.HORIZONTAL)
        m_radioBoxChoices = [u"1", u"2", u"5", u"10", u"20"]
        self.m_radioBox = wx.RadioBox(self, wx.ID_ANY, u"Speed", wx.DefaultPosition, wx.DefaultSize, m_radioBoxChoices,
                                      1, wx.RA_SPECIFY_COLS)
        self.m_radioBox.SetSelection(2)
        middle_tool_box.Add(self.m_radioBox, 0, wx.ALIGN_BOTTOM, 5)
        # gauge1 = PG.PyGauge(self, -1, size=(70, 5), pos=(20, 50), style=wx.GA_HORIZONTAL)
        # gauge1.SetValue(70)
        # # gauge1.SetDrawValue(draw=True, drawPercent=True, font=None, colour=wx.BLACK, formatString=None)
        # gauge1.SetBackgroundColour(wx.WHITE)
        # gauge1.SetBorderColor(wx.BLACK)
        # middle_tool_box.Add(gauge1, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.TOP, 20)

        self.figure_animation = Figure()
        self.axes_animation = self.figure_animation.subplots()
        # self.figure_animation.set_tight_layout(True)

        self.women_img2 = mpimg.imread('Lx_Icons/Women_325x350.png')
        self.show_img = self.axes_animation.imshow(self.women_img2)
        self.canvas_animation = FigureCanvasWxAgg(self, -1, self.figure_animation)
        # self.axes_animation.xticks(np.arange(0,10,1))
        # self.axes_animation.get_xaxis().set_visible(False)
        # self.axes_animation.get_yaxis().set_visible(False)
        # self.axes_animation.set_axis_off()
        # self.figure_animation.subplots_adjust(bottom=0, top=1, left=0, right=1)

        # tight image
        self.axes_animation.set_axis_off()
        self.figure_animation.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        self.axes_animation.margins(0, 0)
        self.figure_animation.gca().xaxis.set_major_locator(plt.NullLocator())
        self.figure_animation.gca().yaxis.set_major_locator(plt.NullLocator())

        middle_tool_box.Add(self.canvas_animation)
        vert_box.Add(middle_tool_box)  # bottom space added
        '''End of middle_tool_box'''

        self.bottomToolbar = wx.ToolBar(self, style=wx.TB_HORIZONTAL | wx.EXPAND | wx.TB_FLAT | wx.TB_NODIVIDER)
        self.bottomToolbar.SetToolBitmapSize(wx.Size(55, 35))
        self.bottomToolbar.SetBackgroundColour('#4086aa')

        for i in range(2):
            self.bottomToolbar.AddStretchableSpace()
        img = wx.Image('Lx_Icons/Ani_Redo_Normal.png').Scale(55, 35, wx.IMAGE_QUALITY_HIGH)
        b_redo_tool = self.bottomToolbar.AddTool(9, 'Animation', wx.Bitmap(img), 'Animate current signal')
        self.Bind(wx.EVT_TOOL, self.on_animate_signal, b_redo_tool)

        for i in range(25):
            self.bottomToolbar.AddStretchableSpace()
        img = wx.Image('Lx_Icons/Ani_Redo_Normal.png').Scale(55, 35, wx.IMAGE_QUALITY_HIGH)
        b_redo_tool = self.bottomToolbar.AddTool(9, 'Animation', wx.Bitmap(img), 'Animate current signal')
        self.Bind(wx.EVT_TOOL, self.on_animate_signal, b_redo_tool)


        # for i in range(2):
        #     self.bottomToolbar.AddStretchableSpace()

        img = wx.Image('Lx_Icons/Ani_Play_Normal.png').Scale(55, 35, wx.IMAGE_QUALITY_HIGH)
        b_play_tool = self.bottomToolbar.AddTool(8, 'Animation', wx.Bitmap(img), 'Animate current signal')
        self.Bind(wx.EVT_TOOL, self.OnClicked, b_play_tool)

        # for i in range(2):
        #     self.bottomToolbar.AddStretchableSpace()

        img = wx.Image('Lx_Icons/Flip.png').Scale(55, 35, wx.IMAGE_QUALITY_HIGH)
        self.b_flip_tool = self.bottomToolbar.AddTool(7, 'Flip', wx.Bitmap(img), 'Flip')
        self.on_flip_click(None, init_flip=True)
        self.Bind(wx.EVT_TOOL, self.on_flip_click, self.b_flip_tool)

        # for i in range(2):
        #     self.bottomToolbar.AddStretchableSpace()
        self.btnFaceImages = ['Lx_Icons/Boy_Face.png', 'Lx_Icons/Girl_Face.png', 'Lx_Icons/Men_Face.png',
                              'Lx_Icons/Women_Face.png']
        self.graphFaceImages = ['Lx_Icons/Boy.png', 'Lx_Icons/Girl.png', 'Lx_Icons/Man.png', 'Lx_Icons/Women.png']
        self.img_index = 0
        img = wx.Image('Lx_Icons/Women_Face.png').Scale(55, 35, wx.IMAGE_QUALITY_HIGH)
        b_face_tool = self.bottomToolbar.AddTool(6, 'Animation', wx.Bitmap(img), 'Animate current signal')
        self.on_face_click(None, init_face=True)
        self.Bind(wx.EVT_TOOL, self.on_face_click, b_face_tool)

        # for i in range(2):
        #     self.bottomToolbar.AddStretchableSpace()

        img = wx.Image('Lx_Icons/Eye.png').Scale(55, 35, wx.IMAGE_QUALITY_HIGH)
        b_eye_tool = self.bottomToolbar.AddTool(5, 'Animation', wx.Bitmap(img), 'Animate current signal')
        self.Bind(wx.EVT_TOOL, self.on_animate_signal, b_eye_tool)



        # for i in range(2):
        #     self.bottomToolbar.AddStretchableSpace()
        img = wx.Image('Lx_Icons/LP.png').Scale(55, 35, wx.IMAGE_QUALITY_HIGH)
        b_lp_tool = self.bottomToolbar.AddTool(4, 'Animation', wx.Bitmap(img), 'Animate current signal')
        self.Bind(wx.EVT_TOOL, self.on_animate_signal, b_lp_tool)


        # for i in range(2):
        #     self.bottomToolbar.AddStretchableSpace()

        img = wx.Image('Lx_Icons/Level.png').Scale(55, 35, wx.IMAGE_QUALITY_HIGH)
        b_level_tool = self.bottomToolbar.AddTool(3, 'Animation', wx.Bitmap(img), 'Animate current signal')
        self.on_level_click(None, init_level=True)
        self.Bind(wx.EVT_TOOL, self.on_level_click, b_level_tool)

        # for i in range(2):
        #     self.bottomToolbar.AddStretchableSpace()

        img = wx.Image('Lx_Icons/Pitch.png').Scale(55, 35, wx.IMAGE_QUALITY_HIGH)
        self.b_pitch_tool = self.bottomToolbar.AddTool(2, 'Pitch', wx.Bitmap(img), 'Pitch')
        self.on_pitch_click(None, init_pitch=True)
        self.Bind(wx.EVT_TOOL, self.on_pitch_click, self.b_pitch_tool)
        # for i in range(2):
        #     self.bottomToolbar.AddStretchableSpace()

        img = wx.Image('Lx_Icons/Keyboard_Normal.png').Scale(55, 35, wx.IMAGE_QUALITY_HIGH)
        b_keyboard_tool = self.bottomToolbar.AddTool(1, 'Animation', wx.Bitmap(img), 'Animate current signal')
        self.Bind(wx.EVT_TOOL, self.on_animate_signal, b_keyboard_tool)

        self.bottomToolbar.Realize()

        vert_box.Add(self.bottomToolbar, 0, wx.EXPAND)
        '''End of bottom_tool_box'''

        self.SetSizer(vert_box)
        self.Refresh()
        self.x = []
        self.y = []
        self.jaw_outline, = self.axes_animation.plot(self.x, self.y, color='#70361b')

    def set_animation_data_plot(self, mat_px, mat_py, K2_POA_pos, h2_POA_pos, time):

        self.mat_px = mat_px  # x axis
        self.mat_py = mat_py  # y axis
        self.K2_POA_pos = K2_POA_pos
        self.h2_POA_pos = h2_POA_pos  # articulation/ movement of red point
        self.time = time

    def set_animation_data_plot_s_op2(self, mat_px, mat_py, K2_POA_pos, h2_POA_pos):
        self.mat_px_s_op2 = mat_px  # x axis
        self.mat_py_s_op2 = mat_py  # y axis
        self.K2_POA_pos_s_op2 = K2_POA_pos
        self.h2_POA_pos_s_op2 = h2_POA_pos  # articulation/ movement of red point

    def OnClicked2(self, e):
        print("ok")
        self.ani = animation.FuncAnimation(self.figure_animation, self.animate_s_op2, init_func=self.init, interval=0.1,
                                           frames=len(self.mat_px_s_op2), repeat=False,
                                           blit=True)

    def OnClicked(self, e):
        print("ok")
        # interval_val = self.time/len(self.mat_px)
        self.ani = animation.FuncAnimation(self.figure_animation, self.animate, init_func=self.init, interval=0.1,
                                           frames=len(self.mat_px), repeat=False,
                                           blit=True)

    def init(self):  # only required for blitting to give a clean slate.
        x = self.mat_px[0]
        y = self.mat_py[0]
        self.jaw_outline.set_data(x, y)
        return self.show_img, self.jaw_outline

    def animate(self, i):
        # pass values, if condition
        # update the data
        x = self.mat_px[i]
        y = self.mat_py[i]
        self.jaw_outline.set_data(x, y)
        poa = self.axes_animation.scatter(self.h2_POA_pos[i], self.K2_POA_pos[i], color='red', s=150)
        jaw_area_fill = self.axes_animation.fill_between(x, y, 0, facecolor=[(254 / 255, 157 / 255, 111 / 255)])

        return self.show_img, self.jaw_outline, jaw_area_fill, poa

    def animate_s_op2(self, i):
        # update the data
        x = self.mat_px_s_op2[i]
        y = self.mat_py_s_op2[i]
        self.jaw_outline.set_data(x, y)
        poa = self.axes_animation.scatter(self.h2_POA_pos_s_op2[i], self.K2_POA_pos_s_op2[i], color='red', s=150)
        jaw_area_fill = self.axes_animation.fill_between(x, y, 0, facecolor=[(254 / 255, 157 / 255, 111 / 255)])

        return self.show_img, self.jaw_outline, jaw_area_fill, poa

    def on_animate_signal(self, e):
        print('Animate signal')

    def on_flip(self, e):

        print('on_flip_signal signal')
        # self.axes_animation = self.figure_animation.gca()
        # self.axes_animation.invert_xaxis()
        self.axes_animation.clear()
        self.show_img = np.fliplr(self.women_img2)
        self.axes_animation.imshow(np.fliplr(self.women_img2))
        self.canvas_animation.draw()
        self.canvas_animation.Refresh()

    def on_flip_click(self, event, init_flip=False):
        if init_flip:
            self.Test_flip_click = False
        else:
            self.Test_flip_click = not self.Test_flip_click
        if self.Test_flip_click:
            self.axes_animation.clear()
            self.axes_animation.set_axis_off()
            self.show_img = np.fliplr(self.women_img2)
            self.axes_animation.imshow(np.fliplr(self.women_img2))
            self.canvas_animation.draw()
            self.canvas_animation.Refresh()
            img = "Lx_Icons/Flip_Reverse.png"
            png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            self.bottomToolbar.SetToolNormalBitmap(id=7, bitmap=png)
        else:
            self.axes_animation.clear()
            self.axes_animation.set_axis_off()
            self.show_img = self.axes_animation.imshow(self.women_img2)
            self.canvas_animation.draw()
            self.canvas_animation.Refresh()
            img = "Lx_Icons/Flip.png"
            png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            self.bottomToolbar.SetToolNormalBitmap(id=7, bitmap=png)

    def on_pitch_click(self, event, init_pitch=False):
        if init_pitch:
            self.Test_pitch_click = False
        else:
            self.Test_pitch_click = not self.Test_pitch_click
        if self.Test_pitch_click:
            pub.sendMessage("PITCH_CHANGE", value=250)

        else:
            pub.sendMessage("PITCH_CHANGE", value=500)

    def on_level_click(self, event, init_level=False):
        if init_level:
            self.Test_level_click = False
        else:
            self.Test_level_click = not self.Test_level_click
        if self.Test_level_click:
            pub.sendMessage("LEVEL_CHANGE", value=60)

        else:
            pub.sendMessage("LEVEL_CHANGE", value=80)

    def on_face_click(self, event, init_face=False):
        if (self.img_index == len(self.btnFaceImages)):
            self.img_index = 0

        img = self.btnFaceImages[self.img_index]
        png = wx.Image(img, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.bottomToolbar.SetToolNormalBitmap(id=6, bitmap=png)
        self.axes_animation.clear()
        self.axes_animation.set_axis_off()
        graph_face = self.graphFaceImages[self.img_index]
        self.women_img2 = mpimg.imread(graph_face)
        self.show_img = self.axes_animation.imshow(self.women_img2)
        self.canvas_animation.draw()
        self.canvas_animation.Refresh()
        self.img_index = self.img_index + 1

