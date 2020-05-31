import tkinter as tk
from framework import tk_utils as tku
import cv2 as cv


class ChoiceVideo(tku.Toplevel):
    def __init__(self, parent):
        super().__init__()
        self.title('choice video')
        self.parent = parent
        # 弹窗界面
        self.setup_UI()

    def setup_UI(self):

        style = {'online_video', 'camera_video'}
        select_win = tk.Frame(self)
        select_win.pack(fill='x')

        self.name = tk.StringVar(value='online_video')
        label = tk.Label(select_win, bg='yellow', width=25, textvar=self.name)
        label.pack()

        for styleItem in style:
            tk.Radiobutton(select_win, text=styleItem, value=styleItem,  variable=self.name).pack()

        tk.Button(select_win, text='确定', command=self.confirm).pack()


    def confirm(self):
        video_path = 'F:\毕业设计\视频\\'+ self.name.get()+'.mp4'
        cap = cv.VideoCapture(video_path)
        # FPS 保存、显示动态视频的信息数量
        FPS = cap.get(cv.CAP_PROP_FPS)
        # print(FPS)
        delay = int(1000 / FPS)

        while cap.isOpened():
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            cv.namedWindow('video preview', 0)
            # cv.resizeWindow('video preview', 1600, 9000)
            cv.imshow('video preview', frame)
            # press q to exit
            if cv.waitKey(delay) == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

        # img_path = 'D:/bysj/optimize/fast-neural-style-video/img/style/' + self.name.get() + '.jpg'
        # self.parent.style_file_name = img_path
        # self.parent.style_img.set_image(img_path)
        # # self.userinfo = [self.name.get()]
        # self.destroy()
        # v = var.get()
        # select_win.destroy()
        # img_path = 'D:/bysj/optimize/fast-neural-style-video/img/style/' + var.get() + '.jpg'
        # print("choice_style_img_path:", img_path)
        # return img_path
