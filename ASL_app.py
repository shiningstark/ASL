from tkinter import *
import argparse
from tkinter import scrolledtext
from PIL import Image, ImageTk
import threading
import time
import cv2
import pandas as pd
import pafy
from yolo import YOLO


def handlerAdaptor(fun, **kwds):
    '''事件处理函数的适配器，相当于中介，那个event是从那里来的呢，我也纳闷，这也许就是python的伟大之处吧'''
    return lambda event, fun=fun, kwds=kwds: fun(event, **kwds)


def onselect(evt, app):
    # Note here that Tkinter passes an event object to onselect()
    w = evt.widget
    index = int(w.curselection()[0])
    app.get_video(index)


class App:
    def __init__(self, windowsTitle):
        # 初始化应用窗口
        self.root = Tk()
        # w, h = self.root.maxsize()
        w, h = 1024, 550
        self.root.geometry("{}x{}".format(w, h))  # 设置窗口的宽和高
        self.root.geometry("+120+120")  # 设置窗口初始位置
        self.root.title(windowsTitle)  # 窗口显示标题
        self.root.resizable(0, 0)  # 窗口禁止拉伸
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)  # 关闭窗口时要执行的程序

        self._init_flag()

        self._init_global_var()  # 初始化一些全局变量

        self._init_dev()  # 初始化摄像头

        self._init_widgets()  # 初始化窗体内的组建

        self.start_core_thread()

        self.root.mainloop()

    # 初始化摄像头
    def _init_dev(self):
        self.cam = cv2.VideoCapture(self.video_path)
        self.asl_video = None
        # 将分辨率设置为 640 X 480
        self.cam.set(3, 640)
        self.cam.set(4, 480)

    # 初始化窗口内的组件
    def _init_widgets(self):
        ###### 主窗口上进行第一层布局  ############

        # 在主窗口上，创建left_fm,并靠左对齐
        self.left_fm = Frame(self.root,
                             bd=1,
                             padx=2,
                             pady=2,
                             bg='#05668D',
                             relief=GROOVE)
        self.left_fm.pack(side=LEFT, fill=BOTH)

        # 在主窗口上创建right_fm,靠左对齐。结果就是与left_fm左右并列
        self.right_fm = Frame(self.root,
                              padx=2,
                              pady=2,
                              bg='#028090',
                              relief=GROOVE)

        self.right_fm.pack(side=RIGHT, fill=BOTH)

        ########  right_fm上进行第二层布局  #############
        # 以下是在right_fm上创建4个frame，自上而下布局

        self.right_listbox_fm = Frame(self.right_fm, bg='#028090', bd=1)  # 用于放置视频中物体的label
        self.right_listbox_fm.pack(side=TOP, fill=BOTH)

        self.asl_fm = Frame(self.right_fm, bg='#028090', bd=1)  # 用于放置视频中物体的label
        self.asl_fm.pack(side=TOP, fill=BOTH)

        self.right_log_fm = Frame(self.right_fm, bg='#028090')  # 放置显示日志控件的
        self.right_log_fm.pack(side=TOP, anchor=W, fill=X)

        self.right_bottom_fm = Frame(self.right_fm, bg='#028090', bd=1)  # 放置按钮的frame
        self.right_bottom_fm.pack(side=BOTTOM, anchor=W, fill=BOTH)

        #### 以下是在各个frame上布放控件，按从左向右，从上到下，从内到外的顺序逐个定义#####

        # 在left_fm上创建video_label, 用于显示视频。
        self.video_label = Label(self.left_fm,
                                 width=680,
                                 height=480,
                                 bg='#05668D',
                                 relief=GROOVE)
        self.video_label.pack(expand=YES)

        # 在right_pic_fm上创建listbox, 用于显示label。
        self.listbox_label = Listbox(self.right_listbox_fm,
                                     selectmode=SINGLE,
                                     width=320)
        self.listbox_label.bind('<<ListboxSelect>>', handlerAdaptor(onselect, app=self))
        self.listbox_label.pack(side=RIGHT)

        # create frame to show asl video
        self.asl_video_label = Label(self.asl_fm,
                                     bg='#05668D',
                                     width=320,
                                     # height=13,
                                     relief=GROOVE
                                     )
        self.asl_video_label.pack(side=RIGHT)

        self.log_text = scrolledtext.ScrolledText(self.right_log_fm,
                                                  bg='#D3D3D3',
                                                  width=43,
                                                  height=8,
                                                  padx=1, pady=1,
                                                  relief=GROOVE)
        self.log_text.pack(side=RIGHT)
        # 在right_bottom_fm上放置按钮控件
        # # 模式切换按钮
        self.refresh_button = Button(self.right_bottom_fm,
                                     text='REFRESH',
                                     width=16,
                                     # height=20,
                                     font=('Arial', 12),
                                     command=self.refresh)
        self.refresh_button.pack(side=RIGHT, fill=BOTH)

        self.model_button = Button(self.right_bottom_fm,
                                   text='START',
                                   width=30,
                                   # height=20,
                                   font=('Arial', 12),
                                   command=self.switch_model)
        self.model_button.pack(side=LEFT, fill=BOTH)

        self.show_log('程序启动完毕，当前处于未识别模式。')

    def _init_flag(self):

        parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

        self.flag = parser.parse_args()

    # 初始化全局变量
    def _init_global_var(self):
        self.video_path = './invideo/testyolo3.flv'
        self.url = None
        self.label = None
        self.yolo = YOLO(**vars(self.flag))
        self.asl_video = None
        self.asl_video_fr_count = 0

        self.current_cv2image = None
        self.current_image = None
        self.detect_data = None
        self.output_image = None

        self.data_path = './data.csv'
        self.data = pd.read_csv(self.data_path)
        self.labels = []
        start_image = cv2.imread('./font/ASL.png')
        self.start_image = cv2.resize(start_image, (320, 200))
        self.asl_frame = self.start_image

        self.model = 2

    '''
    启动核心处理线程:
    一个显示实时视频的线程
    一个显示图片的线程
    '''

    def start_core_thread(self):

        # 视频显示线程
        self.display_video_thread = threading.Thread(target=self.video_loop, args=())
        self.display_video_thread.setDaemon(True)

        self.display_asl_video_thread = threading.Thread(target=self.asl_video_loop, args=())
        self.display_asl_video_thread.setDaemon(True)

        self.display_video_thread.start()
        self.display_asl_video_thread.start()

    def asl_video_loop(self):
        if self.asl_video is not None:
            self.asl_video_fr_count += 1
            if self.asl_video_fr_count == self.asl_video.get(cv2.CAP_PROP_FRAME_COUNT):
                self.asl_video_fr_count = 0
                self.asl_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.asl_frame = cv2.resize(self.asl_frame, (320, 200))
        video_fr_PIL = Image.fromarray(cv2.cvtColor(self.asl_frame, cv2.COLOR_BGR2RGB))

        imgtk = ImageTk.PhotoImage(image=video_fr_PIL)
        self.asl_video_label.imgtk = imgtk
        self.asl_video_label.config(image=imgtk)
        self.root.after(25, self.asl_video_loop)
        # else:
        #     self.show_log("No ASL Video!!")

    # 播放实时视频的方法
    def video_loop(self):
        cam_ok, self.frame = self.cam.read()
        if cam_ok:  # frame captured without any errors

            self.frame = cv2.resize(self.frame, (640, 480))
            # 将BGR格式图片转换为RGB格式
            self.current_cv2image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            # 转换为PIL适用的格式
            self.current_image = Image.fromarray(self.current_cv2image)
            if self.model % 2 == 0:
                self.output_image = self.current_image
            elif self.asl_video is not None:
                video_ok, self.asl_frame = self.asl_video.read()
                self.asl_video_fr_count += 1
                self.detect_data = self.yolo.detect_image(self.current_image)
                self.labels = [self.detect_data[2][0]]
                self.output_image = self.yolo.draw(self.current_image, self.asl_frame, self.detect_data, self.label)
                if self.asl_video_fr_count == self.asl_video.get(cv2.CAP_PROP_FRAME_COUNT):
                    self.asl_video_fr_count = 0
                    self.asl_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                self.detect_data = self.yolo.detect_image(self.current_image)
                labels = self.detect_data[2].tolist()
                self.labels = sorted(set(labels))
                self.output_image = self.yolo.draw(self.current_image, None, self.detect_data, None)
            # 转换为tkinter适用的格式
            imgtk = ImageTk.PhotoImage(image=self.output_image)

            # 让label显示适合在tkinter上显示的图片
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
            self.root.after(25, self.video_loop)
        else:
            self.show_log('Video or Webcam Error!!')

    def show_log(self, logMessage):
        self.log_text.insert(END, time.strftime('%H:%M:%S', time.localtime(time.time())) + ' ' + logMessage + '\n')
        self.log_text.see(END)

    def switch_model(self):
        self.model = self.model % 2 + 1
        self.listbox_label.delete(0, END)
        if self.model % 2 == 0:
            self.asl_video = None
            self.asl_frame = self.start_image
            self.show_log("Recognition Close!!!")
        else:
            self.show_log("Recognition Open!!! Press Refresh and Please Select Object")
            self.show_objectlist()

    def refresh(self):
        self.asl_video = None
        self.asl_frame = self.start_image
        self.show_objectlist()

    def show_objectlist(self):
        self.listbox_label.delete(0, END)
        for idx in self.labels:
            label = self.data.iloc[idx, 0]
            self.listbox_label.insert(END, label)

    def get_video(self, listidx):
        self.asl_video_fr_count = 0
        self.label = self.listbox_label.get(listidx)
        index = self.data[self.data.LABELS.values == self.label].index.tolist()
        self.url = self.data.URLS[index[0]]
        # asl_v = pafy.new(self.url)
        # play = asl_v.getbest()
        if self.url != 'none':
            self.asl_video = cv2.VideoCapture(self.url)
            self.show_log('You selected item %s: "%s"' % (self.label, self.url))
        else:
            self.asl_video = None
            self.show_log('You selected item %s, but we do not find any ASL video' % (self.label))

    # 退出时调用的方法
    def destructor(self):
        """ Destroy the root object and release all resources """
        print(time.strftime('%H:%M:%S', time.localtime(time.time())) + " 正在关闭程序……")
        self.root.destroy()
        self.cam.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application



############## 以上为 App 类的定义 #####################


if __name__ == '__main__':
    print(time.strftime('%H:%M:%S', time.localtime(time.time())) + ' 程序启动')
    display = App('ASL')
