from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np

#把press_time处理一下
f = open('./dataset2/press_time.txt')
data = f.readline()
data = data.split(',')
data.pop()
data1 = [int(i) for i in data]
train_y = np.array(data1).reshape(-1, 1)
np.save('train_y_2.npy', train_y)
print(train_y.shape)

#把图片堆叠起来组成array
# i=0
# train_x = np.zeros((200, 100, 100, 3))
# while i<= 199:
#     a = ndimage.imread("./jpg_2/"+str(i)+"jump.jpg") / 255
#     print(a)
#     train_x[i] = a
#     print('第%d张图片'%(i))
#     i += 1
#
# np.save('train_x_2.npy', train_x)
# plt.imshow(train_x[5])
# plt.show()


# #压缩成jpg
# i=0
# while i <= 200:
#     im = Image.open(r"./dataset2/"+str(i) + "autojump.png")
#     w, h = im.size
#     # 将图片压缩，并截取中间部分，截取后为100*100
#     im = im.resize((108, 192), Image.ANTIALIAS)
#     region = (4, 50, 104, 150)
#     im = im.crop(region)
#     # 转换为jpg
#     bg = Image.new("RGB", im.size, (255, 255, 255))
#     bg.paste(im, im)
#     bg.save(r"./jpg_2/"+str(i)+"jump.jpg")
#     print(i)
#     i += 1

