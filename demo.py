# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
import cv2


class ColorQuantization(object):
    def __init__(self):
        self.capture = cv2.VideoCapture(1)


    def run(self):
        # image from camera
        ret, frame_src = self.capture.read()
        if not ret:
            return 100


        # winner


        # decode


        # update


        # display
        cv2.imshow('source', frame_src)
        #cv2.imshow('quantized', frame_qtz)
        #cv2.imshow('som', frame_som)
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