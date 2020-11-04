import cv2 as cv
import os
from tensorflow import keras #导入模型
import tensorflow as tf
import numpy as np
from PIL import Image,ImageDraw,ImageFont

#读取摄像头视频流
cap = cv.VideoCapture(0)

print('按返回键(Esc)退出程序')
i = 1
a = 1#显示已经保存的图片的数量的变量

#告诉opencv使用人脸识别分类器
#classfier = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
classfier = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')

# 导入模型
model = keras.models.load_model('./model/the_recognization_model.h5')
img_height = 150
img_width = 150
class_names = os.listdir('D:/Code/dataset/recognization')  # 获取各个文件夹名存放为标签列表

# 每隔一秒,保存image到创建的作为预测识别的文件夹里
img_file = 'D:/Code/dataset/test_recognization'
if not os.path.exists(img_file):
    os.mkdir(img_file)  # 不存在则创建文件夹

#opencv添加汉字显示
def cv2ImgAddText(img,text,left,top,textColor = (0,255,0),textSize=20):
    if(isinstance(img,np.ndarray)): #判断是否opencv图片类型
        img = Image.fromarray(cv.cvtColor(img,cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8"
    )
    draw.text((left,top),text,textColor,font = fontText)
    return cv.cvtColor(np.asarray(img),cv.COLOR_RGB2BGR)

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
    #下面的路径需要修改为你自己需要保存的路径
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
    #需要转为灰度,脸部检测的要求
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #人脸检测,1.2和3分别为图片缩放比例和需要检测的有效点数
    faceRects = classfier.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 3,minSize = (32,32))
    #大于0则检测到人脸
    if len(faceRects) > 0:
        #单独框出每一张人脸
        for faceRect in faceRects:
            x,y,w,h = faceRect
            #截取脸部图像提交给模型识别
            image = frame[y-10:y+h+10,x-10:x+w+10]
            #保存图片
            img_path = img_file + "/" + "%d.jpg" % (a)
            image = resize_image(image)#图像预处理
            #灰度保存
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            cv.imwrite(img_path, image_gray)
            #cv.imwrite(img_path, image)
            a += 1  # 自增
            #保存完图片将这张图片导入模型进行识别
            score = predict(img_path)
            #result = "This image most likely to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)],100 * np.max(score))
            #score = np.max(score)
            #设置阈值,置信度accuracy大于98%的在图片上显示对应的标签名,小于98%的在图像上显示识别不成功
            if np.max(score) > 0.98:
                result = class_names[np.argmax(score)]
                cv.rectangle(frame, (x - 15, y - 15), (x + w + 15, y + h + 15), (0, 255, 0), 2)
                print(result)
                #cv.putText(frame, result, (x+20,y-30), cv.FONT_HERSHEY_COMPLEX, 1, (200, 0, 0), 2)
                frame = cv2ImgAddText(frame,result,x+20,y-40,(0,255,0),30)
            #else:
            #cv.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)
            #result = 'Unknown...'
            #print(result)
            #cv.putText(frame, result, (x+20,y-40), cv.FONT_HERSHEY_COMPLEX, 1, (200, 0, 0), 2)
            #frame = cv2ImgAddText(frame, result, x + 20, y - 40, (0, 255, 0), 30)
        i += 1
        if not ret:
            print('视频流无法读取或已结束!')
            break
        #显示视频
        cv.imshow('Camera', frame)
        if cv.waitKey(1) == 27:
            clear() #退出程序清空文件夹
            break
# 释放
cap.release()
cv.destroyAllWindows()
