# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
import sys


class SOM(object):
    def __init__(self, h, w, d):
        self.m = np.random.rand(h, w, d).astype(np.float32)
        self.h = h
        self.w = w
        self.d = d


    def __call__(self, x):
        if not len(x.shape) == 2:
            sys.exit('x.shape must be (batch, dim).')

        # reshape array to compute
        self.x = x.reshape(x.shape[0], 1, 1, x.shape[1]).astype(np.float32)
        m = self.m.reshape((1, ) + self.m.shape)

        # L2 Norm
        norm = self.x - m
        norm = norm ** 2
        norm = np.sum(norm, axis=3)
        norm = norm.reshape(x.shape[0], -1)
        c = np.argmin(norm, axis=1)

        # winner index
        self.c_h = c // self.w
        self.c_w = c %  self.w

        return self.c_h, self.c_w


    def update(self, lr, var):
        # mesh grid
        ws, hs = np.meshgrid(np.arange(self.w), np.arange(self.h))

        # (h, w) -> (b, h, w)
        batch_size = self.c_h.shape[0]
        hs = np.tile(hs.reshape(1, self.h, self.w), (batch_size, 1, 1))
        ws = np.tile(ws.reshape(1, self.h, self.w), (batch_size, 1, 1))

        # distance from winner: (b, 1, 1) - (b, h, w) = (b, h, w)
        d_h = (hs - self.c_h.reshape(-1, 1, 1))
        d_w = (ws - self.c_w.reshape(-1, 1, 1))
        d2 = d_h ** 2 + d_w ** 2

        # gaussian function
        g = np.exp(-d2 / (var ** 2 * 2)).astype(np.float32)
        g = g.reshape(g.shape + (1, ))

        # incrimental
        m = self.m.reshape((1, ) + self.m.shape)
        m_batch = m + (self.x - m) * g * lr

        self.m = np.mean(m_batch, axis=0)


if __name__ == '__main__':
    import cv2
    
    som = SOM(8, 8, 3)

    for i in range(1000):
        x = np.random.rand(50, 3)

        som(x)
        som.update(0.1, 1.0)

        img = cv2.resize(som.m, (300, 300), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('som', img)
        cv2.waitKey(1)