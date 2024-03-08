import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')
import IPython.display as ipd
#
import seaborn as sns


audio_data ="C:/Users/Rhea Ram/Desktop/My_Project/Main/Data/"
Actor = os.listdir(audio_data)
Actor.sort()

#funtion to separate male and female
def get_data(Actor):
    emotion,gender,actor,file_path = [],[],[],[]
    for i in Actor:
        filename = os.listdir(audio_data + i)
        for f in filename:
            part = f.split('.')[0].split('-')
            if not len(part) == 7:
                continue
            emotion.append(int(part[2]))
            actor.append(int(part[6]))
            bg = int(part[6])
            if bg%2 == 0:
                bg = "female"
            else:
                bg = "male"
            gender.append(bg)
            file_path.append(audio_data + i + '/' + f)
    return emotion,gender,actor,file_path

#function to create a record of each emtion for the audio dataset
emotion,gender,actor,file_path = get_data(Actor)
emotion_df = pd.DataFrame(emotion)
emotion_df = emotion_df.replace({1:'neutral',
                                 2:'calm',
                                 3:'happy',
                                 4:'sad',
                                 5:'angry',
                                 6:'fear',
                                 7:'disgust',
                                 8:'surprise'})
data_df = pd.concat([pd.DataFrame(actor),pd.DataFrame(gender),emotion_df,pd.DataFrame(file_path)],axis=1)
data_df.columns = ['actor','gender','emotion','file_path']
data_df.to_csv('Saved_Audio/audio.csv')
print(data_df)

#plotting the csv file
data_df.emotion.value_counts().plot(kind='bar', color='#DAA520')
plt.show()
#displaying the prominent 4 emotions selected for my project

#creating a spectrogram for each emotion of each actor
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


def create_pngs_from_wavs(input_path, output_path):
    print(input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        print('input_file', input_file)
        output_file = os.path.join(output_path, file.replace('.wav', '.png'))
        create_spectrogram(input_file, output_file)


create_pngs_from_wavs('Emotion_New/Angry', 'Spectrogram/Spect_Angry')
create_pngs_from_wavs('Emotion_New/Happy', 'Spectrogram/Spect_Happy')
create_pngs_from_wavs('Emotion_New/Fear', 'Spectrogram/Spect_Fear')
create_pngs_from_wavs('Emotion_New/Sad', 'Spectrogram/Spect_Sad')
#create_pngs_from_wavs('Emotion_New/Calm', 'Spectrogram/Spect_Calm')



from tensorflow.keras.preprocessing import image


def load_images_from_path(path, label):
    images = []
    labels = []

    for file in os.listdir(path):
        images.append(image.img_to_array(image.load_img(os.path.join(path, file), target_size=(224, 224, 3))))
        labels.append((label))

    return images, labels


def show_images(images):
    fig, axes = plt.subplots(1, 8, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i] / 255)


x = []
y = []

#loading spectrogram images for Actor1

images, labels = load_images_from_path('Spectrogram/Spect_Angry', 0)
#show_images(images)

x += images
y += labels

#loading spectrogram images for Actor2

images, labels = load_images_from_path('Spectrogram/Spect_Happy', 1)
#show_images(images)

x += images
y += labels

#loading spectrogram images for Actor3

images, labels = load_images_from_path('Spectrogram/Spect_Fear', 2)
#show_images(images)

x += images
y += labels

#loading spectrogram images for Actor4

images, labels = load_images_from_path('Spectrogram/Spect_Sad', 3)
#show_images(images)

x += images
y += labels





#training and testing the data

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=0)

x_train_norm = np.array(x_train) / 255
x_test_norm = np.array(x_test) / 255

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

#model creation

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#the trained cnn is stored in a variable
hist = model.fit(x_train_norm, y_train_encoded, validation_data=(x_test_norm, y_test_encoded), batch_size=10, epochs=20)
model.save('Audio_1.h5')

print(x_test_norm.shape)
print(y_test_encoded.shape)

#accuracy plotting
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training Accuracy')
plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
plt.title('Training and Validation Accuracy for Model1(Basic CNN)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.plot()
plt.show()


loss, acc = model.evaluate(x_test_norm, y_test_encoded)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()

y_predicted = model.predict(x_test_norm)
print(classification_report(y_test_encoded.argmax(axis=1),y_predicted.argmax(axis=1)))
mat = confusion_matrix(y_test_encoded.argmax(axis=1), y_predicted.argmax(axis=1))
print('mat',mat)
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad' ]

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)

plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.show()

#MobileNetV2 Model


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x_train_norm = preprocess_input(np.array(x_train))
x_test_norm = preprocess_input(np.array(x_test))

train_features = base_model.predict(x_train_norm)
test_features = base_model.predict(x_test_norm)

# %%
"""
Define a neural network to classify features extracted by `MobileNetV2`.
"""

# %%
model2 = Sequential()
model2.add(Flatten(input_shape=train_features.shape[1:]))
model2.add(Dense(1024, activation='relu'))
model2.add(Dense(4, activation='softmax'))
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.summary()
# %%
"""
Train the network with features extracted by `MobileNetV2`.
"""

# %%
hist = model2.fit(train_features, y_train_encoded, validation_data=(test_features, y_test_encoded), batch_size=10, epochs=20)
model2.save('audio12.h5')


acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training Accuracy')
plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
plt.title('Training and Validation Accuracy for Model2 (MobileNetV2)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.plot()
plt.show()

loss, acc = model.evaluate(x_test_norm, y_test_encoded)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()

y_predicted = model2.predict(test_features)
print('y_predicted',y_predicted)
print(classification_report(y_test_encoded.argmax(axis=1),y_predicted.argmax(axis=1)))
mat = confusion_matrix(y_test_encoded.argmax(axis=1), y_predicted.argmax(axis=1))
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad' ]

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='OrRd',
            xticklabels=class_labels,
            yticklabels=class_labels)

plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.show()

