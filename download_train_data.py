#!/usr/bin/env python
# coding=utf-8
# File Name: download_train_data.py
# Author: Lixin Zou
# Mail: zoulixin15@gmail.com
# Created Time: Mon Oct 17 11:58:02 2022

import os

if __name__ == "__main__":
    for i in range(10):
        name = '0' * (5 - len(str(i))) + str(i)
        cmd = "wget " + " https://searchscience.baidu.com/baidu_ultr/train_click/part-" + name + ".gz"
        print(cmd)
        os.system(cmd)
