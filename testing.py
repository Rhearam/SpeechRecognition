import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import librosa.display, os
import matplotlib.pyplot as plt
import warnings
from keras.preprocessing import image
warnings.filterwarnings('ignore')
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import load_model
#base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)
model = load_model('audio12.h5')
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad']

# initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Speech Emotion Recognition')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)


def classify(file_path):
    global label_packed
    print(file_path)
    x = image.load_img('Spectrogram/sample1.png', target_size=(224, 224))
    #image = Image.open('Spectrograms/sample1.png')
    #image = image.resize((224, 224))
    plt.xticks([])
    plt.yticks([])
    #plt.imshow(x)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    y = model.predict(x)
    predictions = y
    print('predictions', predictions)

    print('preds', predictions[0])
    a = predictions[0]
    ind = np.argmax(a)
    print('Prediction:', class_labels[ind])
    result = class_labels[ind]
    print('result', result)
    sign = result
    #sign = 'rakesh'


    #Report end------------
    print(sign)
    label.configure(foreground='#011638', text=sign)


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        create_spectrogram(file_path, 'Spectrogram/sample1.png')
        uploaded = Image.open('Spectrogram/sample1.png')
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="Upload an .wav File", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Speech Emotion Recognition", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()