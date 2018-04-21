from PIL import Image
import numpy as np
i = 0
while True:
    im = Image.open("./datasets/"+str(i)+"autojump.png")
    w, h = im.size
    # 将图片压缩，并截取中间部分，截取后为100*100
    im = im.resize((108, 192), Image.ANTIALIAS)
    region = (4, 50, 104, 150)
    im = im.crop(region)
    # 转换为jpg
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, im)
    file_name = str(i) + "autojump.jpg"
    bg.save(r"./datasets/" + file_name)
    i += 1
# f = open('./datasets/press_time.txt', 'r')
# data = f.readline()
# data = data.split(',')
# data.pop()
# print(len(data))
