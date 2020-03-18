# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
import cv2
import sys

from .som import SOM


class ColorQuantization(object):
    def __init__(self, codebook=(4, 4)):
        if not len(codebook) == 2:
            sys.exit('codebook must be (height, width)')

        self.capture = cv2.VideoCapture(0)
        self.som = SOM(codebook[0], codebook[1], 3)


    def run(self, lr=0.1, var=1.0, img_size=(320, 240)):
        if not len(img_size) == 2:
            sys.exit('img_size must be (width, height)')

        # image from camera
        ret, frame_src = self.capture.read()
        if not ret:
            return 100

        # input -> winner
        frame_src = cv2.resize(frame_src, img_size)
        x = frame_src.reshape(-1, 3)
        x = x.astype(np.float32) / 255.0
        ch, cw = self.som(x)

        # decode
        codebook = self.som.m
        frame_qtz = codebook[ch.reshape(-1, ), cw.reshape(-1, )]
        frame_qtz = frame_qtz.reshape(frame_src.shape[0], frame_src.shape[1], 3)
        frame_qtz = (frame_qtz * 255).astype(np.uint8)

        # update
        self.som.update(lr, var)

        # display
        frame_som = (codebook * 255).astype(np.uint8)
        frame_som = cv2.resize(frame_som, (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('source', frame_src)
        cv2.imshow('quantized', frame_qtz)
        cv2.imshow('som', codebook)
        key = cv2.waitKey(1)

        return key


if __name__ == '__main__':
    color_quantization = ColorQuantization()

    while True:
        key = color_quantization.run()

        if key & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)