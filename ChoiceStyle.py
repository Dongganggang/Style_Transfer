import tkinter as tk
from framework import tk_utils as tku


class ChoiceStyle(tku.Toplevel):
    def __init__(self, parent):
        super().__init__()
        self.title('choice style')
        self.parent = parent
        # 弹窗界面
        self.setup_UI()

    def setup_UI(self):

        style = {'cubist', 'denoised_starry', 'feathers', 'mosaic', 'scream', 'udnie', 'wave', 'ink'}
        select_win = tk.Frame(self)
        select_win.pack(fill='x')

        self.name = tk.StringVar(value='scream')
        label = tk.Label(select_win, bg='yellow', width=25, textvar=self.name)
        label.pack()

        for styleItem in style:
            tk.Radiobutton(select_win, text=styleItem, value=styleItem,  variable=self.name).pack()

        tk.Button(select_win, text='确定', command=self.confirm).pack()


    def confirm(self):
        img_path = 'D:/bysj/optimize/fast-neural-style-video/img/style/' + self.name.get() + '.jpg'
        self.parent.style_file_name = img_path
        self.parent.style_img.set_image(img_path)
        # self.userinfo = [self.name.get()]
        self.destroy()
        # v = var.get()
        # select_win.destroy()
        # img_path = 'D:/bysj/optimize/fast-neural-style-video/img/style/' + var.get() + '.jpg'
        # print("choice_style_img_path:", img_path)
        # return img_path
