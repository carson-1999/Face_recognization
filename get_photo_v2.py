import cv2 as cv
import os

#读取摄像头视频流
cap = cv.VideoCapture(0)

#创建保存的文件夹
#输入要录入的姓名作为数据集的文件夹名字
name = input('请输入照片数据录入人的姓名:(输入英文名) ')
number = input('请输入照片数据录入数量: ')
 #这里以我的照片存放目录D:/Code/dataset/recognization/为例，这里你可以更改为你的存放目录
img_file = 'D:/Code/dataset/recognization/%s' % (name)
if not os.path.exists(img_file):
        os.mkdir(img_file)  # 不存在则创建文件夹

print('按返回键(Esc)退出程序')
i = 1
a = 1#显示已经保存的图片的数量的变量

#告诉opencv使用人脸识别分类器
#classfier = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
classfier = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')

IMAGE_SIZE = 150

def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    # 获取图像尺寸
    h, w, _ = image.shape
    # 对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)
    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    # RGB颜色
    BLACK = [0, 0, 0]
    # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=BLACK)
    # 调整图像大小并返回
    return cv.resize(constant, (height, width))

while cap.isOpened():
    ret,frame = cap.read()
    # print(frame.shape) #本摄像头是(480,640,3),即(rows,cols,channels)
    #需要转为灰度,脸部检测的要求
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #人脸检测,1.2和3分别为图片缩放比例和需要检测的有效点数
    faceRects = classfier.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 3,minSize = (32,32))
    #大于0则检测到人脸
    if len(faceRects) > 0:
        #单独框出每一张人脸
        for faceRect in faceRects:
            x,y,w,h = faceRect
            image = frame[y-10:y+h+10,x-10:x+w+10]
            img_path = img_file + "/" + "%d.jpg" % (a)
            image = resize_image(image)#图像预处理
            #灰度保存图片
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            cv.imwrite(img_path, image_gray)
            #cv.imwrite(img_path, image)
            a += 1  # 自增
        i += 1
        if(a==int(number)+1): #当收集到500张照片退出
            print('已收集到%s张照片'%(number))
            break
        if not ret:
            print('视频流无法读取或已结束!')
            break
        #显示视频
        cv.imshow('Camera', frame)
        if cv.waitKey(1) == 27:
            break
# 释放
cap.release()
cv.destroyAllWindows()