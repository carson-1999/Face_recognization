from tensorflow import keras
import  tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from matplotlib import pyplot as plt
import numpy as np

data_dir = pathlib.Path('D://Code//dataset//recognization')
data_dir = pathlib.Path(data_dir) #下载的文件路径转为文件具体路径
img_count = len(list(data_dir.glob('*/*.jpg')))
#print(img_count)  #共有25000张图片

"""创建一个数据集"""
#定义参数
batch_size=32
img_height = 150
img_width = 150

#用80%的图片作为训练的数据集
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.25,
    subset = "training",
    seed = 123,
    image_size=(img_height,img_width),
    batch_size = batch_size
)
#创建验证模型的数据集  validation 生效,批准,验证
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.25,
    subset = "validation",
    seed=123,
    image_size=(img_height,img_width),
    batch_size=batch_size
)
#可以在训练数据集的class_names属性中找到类名,它们与按字母顺序排列的目录名相对应。
class_names = train_ds.class_names
print(class_names)
#配置数据集
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#数据扩充
data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal",input_shape=(img_height,img_width,3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1)
])

num_classes = len(class_names)#分类的标签数目

#模型创建
model = Sequential([
    data_augmentation,
    #数据标准化
    layers.experimental.preprocessing.Rescaling(1./255),
    #卷积神经网络
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    #添加Dropout
    layers.Dropout(0.2),
    #常规三层
    layers.Flatten(),
    layers.Dense(480,activation='relu'),
    layers.Dense(num_classes)
    ])
# 编译模型
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
#model.summary()
#训练模型
epochs = 6
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs = epochs
)
#save,保存模型
model.save('./model/the_recognization_model.h5')
#可视化模型训练结果
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range,acc,label = 'Training Accuracy')
plt.plot(epochs_range,val_acc,label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range,loss,label = 'Training loss')
plt.plot(epochs_range,val_loss,label = 'Validation loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')
plt.show()


#animal_url = 'https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1603639888012&di=7bdf445d47d8763491e6dfc090fb5ac3&imgtype=0&src=http%3A%2F%2F5b0988e595225.cdn.sohucs.com%2Fq_70%2Cc_zoom%2Cw_640%2Fimages%2F20181115%2Fa1d4636a65064eabb12ddce624d3e8c7.jpeg'
#animal_path = tf.keras.utils.get_file('test',origin=animal_url)
animal_path = '1.jpg'

img = keras.preprocessing.image.load_img(
    animal_path,target_size=(img_height,img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array,0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(
    "This image most likely to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)],100 * np.max(score))
)
