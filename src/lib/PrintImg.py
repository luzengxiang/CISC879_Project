#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:55:58 2017

@author: zengxiang
"""

import numpy as np
import os
import matplotlib.pyplot
from lib import DataInput

def save_images(path,outpath):
    savepath= os.path.splitext(outpath)[0]+'/'
    if os.path.exists(savepath)==False:
        os.mkdir(savepath)
    data = DataInput.read_data(path)
    for i in range(data.shape[-1]):
        img = np.flipud(data[:,:,i].transpose())
        matplotlib.pyplot.imsave(savepath+str(i)+'.png',img)
