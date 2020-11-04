import cv2 as cv
import os
from tensorflow import keras #导入模型
import tensorflow as tf
import numpy as np

#读取摄像头视频流
cap = cv.VideoCapture(0)

print('按返回键(Esc)退出程序')
i = 1#计算有有几帧图片,即循环执行了几次的变量
timeF = 18 #程序i自加操作大约为0.1秒,这里设置为10,相当于每1秒获取保存一次子图
a = 1#显示已经保存的图片的数量的变量


img_file = 'D:/Code/dataset/test_recognization'
if not os.path.exists(img_file):
    os.mkdir(img_file)  # 不存在则创建文件夹

# 导入模型
model = keras.models.load_model('./model/the_recognization_model.h5')
img_height = 150
img_width = 150
class_names = os.listdir('D:/Code/dataset/recognization')  # 获取各个文件夹名存放为标签列表

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

#传图片给模型进行预测
def predict(img_path):
    img = keras.preprocessing.image.load_img(
        img_path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return score


#退出程序便清空保存识别图片的文件夹
def clear():
    #下面的路径需要保存为你需要保存的路径
    img_path = 'D:/Code/dataset/test_recognization'
    img_files = os.listdir(img_path)
    for img_file in img_files:
        img = os.path.join(img_path,img_file)
        if os.path.isfile(img):
            os.remove(img)
    print('文件夹已清空')

while cap.isOpened():
    ret,frame = cap.read()
    # print(frame.shape) #本摄像头是(480,640,3),即(rows,cols,channels)
    size = frame.shape
    height = size[0]
    width = size[1]
    x = int(width / 3)
    y = int(height / 4)
    # 画出人脸框,绿色框
    color = (0, 255, 0)
    cv.rectangle(frame, (x, y), (x + 260, y + 260), color, 2)
    #写入文字
    text = 'Recognization area: '
    cv.putText(frame, text, (x - 50, y - 10), cv.FONT_HERSHEY_COMPLEX, 1, (200, 0, 0), 2)
    # 获取子图，即框区域作为ROI区域
    roi = frame[y:y + 260, x:x + 260]
    if (i % timeF == 0): # 每隔一秒,保存roi到创建的作为预测识别的文件夹里
        #保存图片
        img_path = img_file + "/" + "%d.jpg" % (a)
        #灰度转化保存
        roi = resize_image(roi)
        roi_gray = cv.cvtColor(roi,cv.COLOR_BGR2GRAY)
        cv.imwrite(img_path, roi_gray)
        a += 1  # 自增
        #保存完图片将这张图片导入模型进行识别
        score = predict(img_path)
        #result = "This image most likely to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)],100 * np.max(score))
        #score = np.max(score)
        #设置阈值,置信度accuracy大于98%的在图片上显示对应的标签名,小于98%的在图像上显示识别不成功
  #阈值设置   #if np.max(score) >= 0.98:
        result = class_names[np.argmax(score)]
        print(result)
        cv.putText(frame, result, (x+10,y+290), cv.FONT_HERSHEY_COMPLEX, 1, (200, 0, 0), 2)
        #result = 'Identifying......'
        #print(result)
        #cv.putText(frame, result, (x+10,y+290), cv.FONT_HERSHEY_COMPLEX, 1, (200, 0, 0), 2)
    i += 1
    if not ret:
        print('视频流无法读取或已结束!')
        break
    #显示视频
    cv.imshow('Camera', frame)
    if cv.waitKey(1) == 27:
        #清空文件夹后退出
        clear()
        break
# 释放
cap.release()
cv.destroyAllWindows()
