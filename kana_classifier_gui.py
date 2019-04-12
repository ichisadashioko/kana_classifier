from __future__ import absolute_import, division, print_function

import os

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

tf.enable_eager_execution()

import kivy

from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import *
from kivy.app import App
from kivy.core.window import Window


kana_model = None
label_dict = None


def merge_sorted_arrays(a, b):

    retval = []

    if not isinstance(a, list):
        a = a.tolist()

    if not isinstance(b, list):
        b = b.tolist()

    while len(a) > 0 and len(b) > 0:
        _a = a[0][1]
        _b = b[0][1]

        if _a > _b:
            retval.append(a.pop(0))
        else:
            retval.append(b.pop(0))

    retval.extend(a)
    retval.extend(b)

    return retval


def merge_sort(arr):

    if len(arr) > 1:
        half_pts = int(len(arr) / 2)

        _a = arr[:half_pts]
        _b = arr[half_pts:]

        _a = merge_sort(_a)
        _b = merge_sort(_b)

        return merge_sorted_arrays(_a, _b)

    else:
        return arr


def preprocess_image(img):
    resize_img = cv2.resize(img, (64, 64), cv2.INTER_CUBIC)
    scale_value = resize_img.astype(np.float) / 255.0
    reshape_img = np.reshape(scale_value, (1, 64, 64, 1))

    return reshape_img


def predict_kana(img):
    hist = model.predict(img)
    result = hist[0]

    idx_result = [[idx, value] for idx, value in enumerate(result)]
    sort_result = merge_sort(idx_result)
    map_result = [[label_dict[idx]['kana'], c] for idx, c in sort_result]

    return map_result

def predict_pipeline(img):
    if img is None:
        return None

    img = preprocess_image(img)
    result = predict_kana(img)
    return result

class Pen():

    def __init__(self):
        self.down = False
        self.touch_id = None
        self.pos = None

    def up(self):
        self.down = False
        self.touch_id = None
        self.pos = None


class Touch(Widget):

    def __init__(self, **kwargs):
        super(Touch, self).__init__(**kwargs)

        self._keyboard = Window.request_keyboard(
            self._keyboard_closed,
            self, 'text'
        )
        self._keyboard.bind(
            on_key_down=self._on_keyboard_down,
            on_key_up=self._on_keyboard_up
        )

        self.pen = Pen()
        self.VERBOSE = 1

    def get_canvas_image(self):
        filename = 'temp.png'
        success = self.export_to_png(filename=filename)
        if success:
            img = cv2.imread(filename, 0)
            return img
        else:
            print('Failed to export_to_png')
            return None
            pass

    def _keyboard_closed(self):
        print('My keyboard have been closed!')
        # self._keyboard.unbind(
        #     on_key_down=self._on_keyboard_down,
        #     on_key_up=self._on_keyboard_up
        # )
        # self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'shift':
            self.pen.down = True
            # Window.show_cursor = False
            # print('Pen down')

    def _on_keyboard_up(self, keyboard, keycode):
        if keycode[1] == 'shift':
            self.pen.up()
            # Window.show_cursor = True
            # print('Pen up')
        elif keycode[1] == 'c':
            with self.canvas:
                Color(0, 0, 0)
                Rectangle(pos=(0, 0), size=self.size)

        elif keycode[1] == 's':
            filename = 'temp.png'
            success = self.export_to_png(filename=filename)
            if success:
                img = cv2.imread(filename, 0)
                print(img.shape)
                plt.imshow(img)
                plt.show()
            else:
                print('Failed to export_to_png')
                pass

    def on_touch_down(self, touch):
        # print('Touch down', touch)

        if self.pen.down:
            if self.pen.touch_id == None:
                self.pen.touch_id = touch.id
                self.pen.pos = touch.pos

                # print('Pen down', self.pen.touch_id, type(self.pen.touch_id))
        else:
            # print('[Touch down] pen is not down')
            pass

    def on_touch_move(self, touch):
        if self.pen.down:
            if self.pen.touch_id == touch.id:

                pts = [
                    self.pen.pos[0],
                    self.pen.pos[1],
                    touch.pos[0],
                    touch.pos[1]
                ]
                with self.canvas:
                    Color(1, 1, 1)
                    Line(
                        points=pts,
                        width=10
                    )

                self.pen.pos = touch.pos

    def on_touch_up(self, touch):
        if self.pen.down:
            if self.pen.touch_id == touch.id:
                self.pen.touch_id = None
                self.pen.pos = None
                # print('Pen up', self.pen.touch_id, type(self.pen.touch_id))
                img = self.get_canvas_image()
                result = predict_pipeline(img)
                if not result is None:
                    # print(type(result), len(result))
                    for i in range(10):
                        print('{}: {:.5f}%'.format(result[i][0], result[i][1]*100))
                    print('='*16)

class KanaApp(App):
    def build(self):
        return Touch()


if __name__ == "__main__":
    model = tf.keras.models.load_model('kana_model.h5')

    label_dict_df = pd.read_csv('label_dict.csv', keep_default_na=False)
    label_dict = label_dict_df.to_dict(orient='index')

    KanaApp().run()
