""" MNIST veri setini kullanacagiz
Rakamlama: 0-9 arasi rakamlardan toplamda 10 sinif var
28*28 piksel boyutunda resimlerden olusuyor
grayscale resimler(siyah-beyaz)
60000 adet egitim ve 10000 adet test verimiz var
amacimiz ann ile bu resimleri tanimlamak veya siniflandirmak

Image Proccesing;
Histogram esitleme:Konstrat iyilestirme
Gaussian bulur;Gurultuyu azaltma
Canny edge detection:Kenar Tespiti

ANN(Artifical Neural Network) ile MNIST veri setini siniflandiracagiz

libraries:
Tensorflow;Keras ile ann model olusturma ve egitim
Matplotlib;Gorsellestirme 
cv2;Opencv ile image processing
"""

#import libraries
import cv2 #opencv
import numpy as np
import matplotlib.pyplot as plt #gorsellestirme icin

from tensorflow.keras.datasets import mnist #MNIST veri seti
from tensorflow.keras.models import Sequential #ANN modeli icin
from tensorflow.keras.layers import Dense, Flatten,Dropout
from tensorflow.keras.optimizers import Adam #optimizer

#load MNIST data set
(x_train,y_train), (x_test,y_test)=mnist.load_data()
print(f"x_train shape:{x_train.shape}")
print(f"y_train shape:{y_train.shape}")

"""x_train shape:(60000, 28, 28)
y_train shape:(60000,)
"""




#image preprocessing
img=x_train[0]#ilk resmi al

stages={"original":img}

#histogram esitleme
eq=cv2.equalizeHist(img)#histogram esitleme
stages["Histogram esitleme"]=eq

#Gaussian blur:Gurultuyu Azaltma
blur=cv2.GaussianBlur(eq,(5,5),0)
stages["Gaussian blur"]=blur

#canny ile kenar tespiti
edges=cv2.Canny(blur,50,150) #kenar tespiti
stages["canny kenarlari"]=edges

#gorsellestirme
fig, axes = plt.subplots(2,2,figsize=(6,6))
axes = axes.flat
for ax,(title,im)in zip(axes,stages.items()):
    ax.imshow(im, cmap = "gray")
    ax.set_title(title)
    ax.axis("off")


plt.suptitle("Mnist Image Processing Stanges")
plt.tight_layout()
plt.show()

    
"""
    histogram esitleme
    gaussian blur
    canny ile kenar tespiti
    flattering:28*28 boyutundan 784 boyutuna cevirme
    normalizasyon:0-255 arasindan 0-1 arasina cevirme
"""
   # preprocessing fonksıyonu
def preprocess_image(img):
    img_eq = cv2.equalizeHist(img)
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
    img_edges = cv2.Canny(img_blur, 50, 150)
    return img_edges.flatten() / 255.0

# Verileri dönüştürme (x_train ve y_train yerine ön işlenmiş verileri hazırlıyoruz)
num_train = 10000
num_test = 2000

x_train_proc = np.array([preprocess_image(img) for img in x_train[:num_train]])
y_train_proc = y_train[:num_train]

x_test_proc = np.array([preprocess_image(img) for img in x_test[:num_test]])
y_test_proc = y_test[:num_test]
####ann model cretion
model=Sequential([
    Dense(128,activation="relu",input_shape=(784,)),#ilk katman,128 noron 28*28=784 boyutunda
    Dropout(0.5),#dropout katmani,overfitting'i azaltmak icin %50 dropout
    Dense(64,activation="relu"),#ikinci katman,64 noron
    Dense(10,activation="softmax")#cikis katmani,10 noron (0-9 rakamlari icin)
])

#compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),#optimizer
    loss="sparse_categorical_crossentropy",#kayip fonksiyonu
    metrics=["accuracy"] #metrikler 
)
print(model.summary())

# ann model training 
history = model.fit(
    x_train_proc, y_train_proc,
    validation_data=(x_test_proc, y_test_proc),
    epochs=50,
    batch_size=32,
    verbose=2
)

#evaluate model performance 
test_loss, test_acc = model.evaluate(x_test_proc,y_test_proc)
print(f"Test loss: {test_loss:.4f},Test accuracy:{test_acc:.4f}")

#plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["loss"],label="training loss")
plt.plot(history.history["val_loss"],label="Validation loss")
plt.title("loss")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],label="training accuracy")
plt.plot(history.history["val_accuracy"],label="Validation accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

