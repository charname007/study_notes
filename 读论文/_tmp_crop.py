# -*- coding: utf-8 -*-
from PIL import Image
im = Image.open(r"E:\学习笔记\读论文\_tmp_charseq_p3-03.png")
print("size", im.size)
box = (92, 138, 1330, 1798)
im.crop(box).save(r"E:\学习笔记\读论文\ChAR-seq 方法总图 Fig1.png")
print("saved", box)
