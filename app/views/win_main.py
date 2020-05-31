# coding:utf-8
# -----------------------------------------------------------------------------
# 所有窗体，以 WinBase() 为基类,  WinBase.win 为 tk 的源组件；加一层，是为了防止命令冲突

# 引用时，用 parent, parent.win
# 扩展子组件，返回的实际的控件类型，如 self.widget_label, self.widget_frame
# 组件的容器，一律取 master
# label, button , 可通过 style name 来进行设置和渲染
# -----------------------------------------------------------------------------

from __future__ import print_function
import tensorflow as tf
from preprocessing import preprocessing_factory
import reader
import model
import time
import os

import numpy as np
from keras.applications import vgg19

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

import os
import tkinter as tk
from tkinter import ttk
import tensorflow as tf

from framework import tk_utils as tku
from framework import utils
from app import my

from app.my import session
from app.my import img
from app.my import Color
from app.my import String
from app.my import Font

from ChoiceStyle import ChoiceStyle
from ChoiceVideo import ChoiceVideo


import cv2 as cv
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import tkinter


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 设置 GPU 的占用率，以免全部占用
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print("完成！！！！！！！！！！！")
workspace_dir = "workspace"
result_prefix = 'style_transfer_'
iterations = 5

style_weight = 1.
content_weight = 0.025


# 主应用程序界面
class App(tku.WinBase):
# class App(tk.Tk):
    target_file_name = ""
    style_file_name = ""
    generated_file = ""


    def __init__(self):
        tku.WinBase.__init__(self)
        # super().__init__()

        self.title = String.r_app_title
        self.set_size(1024, 768)
        self.set_icon(img("Money.ico"))


        self.lay_body()

    def show_conImage(self):
        r_img = mpimg.imread(self.target_file_name)

        plt.imshow(r_img)  # 显示图片
        plt.axis('off')  # 不显示坐标轴
        plt.title("content image")
        plt.show()

    def show_genImage(self):
        if self.generated_file == "":
            tkinter.messagebox.showerror('错误', '图片不存在')
        else:
            r_img = mpimg.imread(self.generated_file)  # 读取和代码处于同一目录下的 lena.png
            # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
            # lena.shape  # (512, 512, 3)

            plt.imshow(r_img)  # 显示图片
            plt.axis('off')  # 不显示坐标轴
            plt.title("generated image")
            plt.show()

    def lay_body(self):
        self.lay_title(self.win).pack(fill=tk.X)

        self.lay_main(self.win).pack(expand=tk.YES, fill=tk.BOTH)

        self.lay_bottom(self.win).pack(fill=tk.X)

        return self.win

    def lay_title(self, parent):
        """ 标题栏 """
        frame = tk.Frame(parent, bg="black")

        def _label(_frame, text, size=12, bold=False):
            return tku.label(_frame, text, size=size, bold=bold, bg="black", fg="white")

        def _button(_frame, text, size=12, bold=False, width=12, command=None):  # bg = DarkSlateGray
            return tk.Button(_frame, text=text, bg="black", fg="white",
                             width=width, height=2, font=tku.font(size=size, bold=bold),
                             relief=tk.FLAT, command=command)

        _label(frame, String.r_app_title, 16, True).pack(side=tk.LEFT, padx=10)
        _label(frame, "").pack(side=tk.LEFT, padx=50)  # 用于布局的空字符串
        _label(frame, "").pack(side=tk.RIGHT, padx=5)
        _button(frame, "退出", width=8, command=self.do_close).pack(side=tk.RIGHT, padx=15)
        tku.image_label(frame, img("user.png"), 40, 40, False).pack(side=tk.RIGHT)

        return frame

    def lay_bottom(self, parent):
        """ 窗体最下面留空白 """
        frame = tk.Frame(parent, height=10, bg="whitesmoke")
        frame.propagate(True)
        return frame

    def lay_main(self, parent):
        frame = tk.Frame(parent, bg=Color.r_background)

        self.lay_org_image(frame).pack(side=tk.LEFT, fill=tk.Y)
        self.lay_result_image(frame).pack(expand=tk.YES, fill=tk.BOTH, padx=5, pady=5)

        return frame


    def lay_org_image(self, parent):
        frame = tk.Frame(parent, bg=Color.r_whitesmoke, width=350)

        self.org_img = tku.ImageLabel(frame, width=300, height=300)
        self.org_img.pack(side=tk.TOP, fill=tk.Y, padx=10, pady=5)
        self.org_img.set_image(img("mai.jpg"))
        self.target_file_name = img("mai.jpg")
        menu_org = tk.Menu(self.org_img, tearoff=0)
        menu_org.add_command(label="显示大图", command=self.show_conImage)
        menu_org.add_command(label="更改图片", command=self.do_choice_image)

        def popupmenu(event):
            menu_org.post(event.x_root, event.y_root)

        self.org_img.bind('<Button-1>', self.do_choice_image)
        self.org_img.bind('<Button-3>', popupmenu)


        self.style_img = tku.ImageLabel(frame, width=300, height=300)
        self.style_img.pack(side=tk.TOP, padx=10, pady=5)
        self.style_img.set_image(img("scream.jpg"))
        self.style_img.bind('<Button-1>', self.do_choice_style_image)
        self.style_file_name = img("scream.jpg")  # 设置缺省风格格式文件

        # tk.Button(frame, text=" 目标图片 ", bg="LightBlue", font=Font.r_normal, command=self.do_choice_image) \
        #     .pack(side=tk.LEFT, fill=tk.X, padx=20)
        # tk.Button(frame, text=" 风格图片 ", bg="LightYellow", font=Font.r_normal, command=self.do_choice_style_image) \
        #     .pack(side=tk.LEFT, fill=tk.X)
        # tk.Button(frame, text="开始转换", bg="LightGreen", font=Font.r_medium_title, command=self.do_transfer) \
        #     .pack(side=tk.RIGHT, fill=tk.X, padx=20)

        tk.Button(frame, text=" 内容图片 ", bg="LightBlue", font=Font.r_normal, command=self.do_choice_image) \
            .pack(side=tk.LEFT, fill=tk.X, padx=1)
        tk.Button(frame, text=" 风格图片 ", bg="LightYellow", font=Font.r_normal, command=self.do_choice_style_image) \
            .pack(side=tk.LEFT, fill=tk.X, padx=1)
        tk.Button(frame, text=" 视频效果", bg="LightBlue", font=Font.r_normal, command=self.do_play_video) \
            .pack(side=tk.LEFT, fill=tk.X, padx=1)
        tk.Button(frame, text="开始转换", bg="LightGreen", font=Font.r_normal, command=self.do_transfer) \
            .pack(side=tk.LEFT, fill=tk.X, padx=1)



        frame.propagate(True)
        frame.pack_propagate(0)
        return frame

    def lay_result_image(self, parent):
        frame = tk.Frame(parent, bg=Color.r_white)

        fra_result = tk.Frame(frame, bg=Color.r_white)
        fra_result.pack(expand=tk.YES, fill=tk.BOTH, pady=5)
        fra_result.pack_propagate(0)

        self.final_img = tku.ImageLabel(fra_result, width=460, height=460)
        self.final_img.pack(side='top', padx=10, pady=10)
        self.final_img.set_image(img("style_flower.png"))
        # self.final_img.bind('<Button-1>', self.do_browser_workspace)
        menu_res = tk.Menu(self.final_img, tearoff=0)
        menu_res.add_command(label="显示大图", command=self.show_genImage)

        def popupmenu(event):
            menu_res.post(event.x_root, event.y_root)

        self.final_img.bind('<Button-1>', self.do_browser_workspace)
        self.final_img.bind('<Button-3>', popupmenu)

        tk.Button(fra_result, text=" 图片生成目录 ", bg="LightBlue", font=Font.r_small_content,
                  command=self.do_browser_workspace) \
            .pack(side=tk.BOTTOM, anchor='se', padx=10)



        return frame

    # -------------------------------------------------------------------------
    def do_close(self):
        """ 关闭应用程序 """
        msg = String.r_exit_system
        if tku.show_confirm(msg):
            self.close()

    def do_choice_image(self, *args):
        """ 选择目标图片 """
        img_path = tku.ask_for_filename()
        if img_path is not None and img_path != "":
            self.target_file_name = img_path
            self.org_img.set_image(img_path)



    def do_choice_style_image(self, *args):
        """ 选择风格图片 """
        img_style = ChoiceStyle(self)


    def do_browser_workspace(self, *args):
        """ 浏览新生成的图片 """
        if os.path.exists(workspace_dir):
            os.startfile(workspace_dir)

    def do_play_video(self, *args):
        """播放迁移效果视频"""
        ChoiceVideo(self)





    def do_transfer(self):
        tf.logging.set_verbosity(tf.logging.INFO)

        # 获得目标图片，风格图片
        target_image_path = self.target_file_name
        style_reference_image_path = self.style_file_name

        if not os.path.exists(target_image_path):
            tku.show_message("请先选择目标图片文件！")
            return

        if not os.path.exists(style_reference_image_path):
            tku.show_message("请先选择风格图片文件！")
            return

        utils.create_folder(workspace_dir)


        self.final_img.set_image(target_image_path)
        self.refresh()

        # Get image's height and width.
        height = 0
        width = 0
        with open(target_image_path, 'rb') as img:
            with tf.Session().as_default() as sess:
                # if FLAGS.image_file.lower().endswith('png'):
                if target_image_path.lower().endswith('png'):
                    image = sess.run(tf.image.decode_png(img.read()))
                else:
                    image = sess.run(tf.image.decode_jpeg(img.read()))
                height = image.shape[0]
                width = image.shape[1]
        tf.logging.info('Image size: %dx%d' % (width, height))

        with tf.Graph().as_default():
            with tf.Session().as_default() as sess:
                # Read image data.
                image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                    'vgg_16',
                    is_training=False)

                # image = reader.get_image(target_image_path, int(height/2.5), int(width/2.5), image_preprocessing_fn)  # gpu
                image = reader.get_image(target_image_path, height, width, image_preprocessing_fn)  # cpu

                # Add batch dimension
                image = tf.expand_dims(image, 0)

                generated = model.net(image, training=False)
                generated = tf.cast(generated, tf.uint8)

                # Remove batch dimension
                generated = tf.squeeze(generated, [0])

                # Restore model variables.
                saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

                filepath, tempfilename = os.path.split(style_reference_image_path)
                style_name, extension = os.path.splitext(tempfilename)
                model_name = 'models/' + style_name + '.ckpt-done'
                saver.restore(sess, model_name)

                self.generated_file = os.path.join(workspace_dir, result_prefix + utils.strftime(False) + ".jpg")

                # Generate and write image data to file.
                with open(self.generated_file, 'wb') as img:
                    start_time = time.time()
                    img.write(sess.run(tf.image.encode_jpeg(generated)))
                    end_time = time.time()
                    tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

                    tf.logging.info('Done. Please check %s.' % self.generated_file)
                self.final_img.set_image(self.generated_file)
