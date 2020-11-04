import cv2 as cv
import os #路径操作依赖库

#读取摄像头视频流
cap = cv.VideoCapture(0)
#输入要录入的姓名作为数据集的文件夹名字
name = input('请输入照片数据录入人的姓名:(输入英文名) ')
number = input('请输入需要采集的照片数量: ')
print('按返回键(Esc)退出程序')
i = 1
a = 1#显示已经保存的图片的数量的变量
timeF = 12 #程序i自加操作大约为0.1秒,这里设置为10,相当于每1秒获取保存一次子图

while cap.isOpened():
    ret,frame = cap.read()
    #print(frame.shape) #本摄像头是(480,640,3),即(rows,cols,channels)
    size = frame.shape
    height = size[0]
    width = size[1]
    x = int(width/3)
    y = int(height/4)
    # 画出人脸框,绿色框
    color = (0, 255, 0)
    cv.rectangle(frame,(x,y),(x+260,y+260),color,2)
    #获取子图，即框区域作为ROI区域
    roi = frame[y:y+260,x:x+260]
    #每隔一秒,保存roi到创建的文件夹里
    img_file = 'D:/Code/dataset/recognization/%s'%(name) #这里是我存放的目录,注意修改为你自己需要存放的目录
    if not os.path.exists(img_file):
        os.mkdir(img_file)#不存在则创建文件夹
    if (i%timeF == 0):
        #转换为灰度图
        gray = cv.cvtColor(roi,cv.COLOR_BGR2GRAY)
        cv.imwrite(img_file+"/"+"%d.jpg"%(a),gray)
        # 显示文字在框上
        text = "The total number"+str(a)
        cv.putText(frame, text, (x - 80, y - 10), cv.FONT_HERSHEY_COMPLEX, 1, (200, 0, 0), 1)
        a += 1 #自增
    i+=1#每几帧保存图片的变量
    if(a==int(number)+1):
        print('已收集到%s张照片'%number)
        break
    if not ret:
        print('视频流无法读取或已结束!')
        break

    cv.imshow('Camera',frame)
    if cv.waitKey(1) == 27: #按esc键退出
        break
#释放
cap.release()
cv.destroyAllWindows()
