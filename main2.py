import numpy as np

max_pad_len = 431


def prepare_dataset(csv_file):
    plt.figure(figsize=(12, 4))
    data, sample_rate = librosa.load(filename)
    _ = librosa.display.waveplot(data, sr=sample_rate)
    ipd.Audio(filename)

    metadata = pd.read_csv('sounds/sounds/labels.csv')
    print(metadata.head())

    print(metadata.shape)

    # wavfilehelper = WavFileHelper()

    audiodata = []
    features = []

    filenames = metadata['fileName'].values
    filelabels = metadata['fileLabel'].values

    for index, row in metadata.iterrows():
        file_name = os.path.join(os.path.abspath("..\\untitled5\\sounds\\sounds"), filenames[index])
        class_label = filelabels[index]
        data = extract_features(file_name)

        features.append([data, class_label])

    # Convert into a Panda dataframe
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(data, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()

    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    m_slaney = librosa.feature.mfcc(y=audio, sr=sample_rate, dct_type=2)
    m_htk = librosa.feature.mfcc(y=audio, sr=sample_rate, dct_type=3)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(m_slaney, x_axis='time')
    plt.title('RASTAMAT / Auditory toolbox (dct_type=2)')
    plt.colorbar()
    plt.subplot(2, 1, 2)
    librosa.display.specshow(m_htk, x_axis='time')
    plt.title('HTK-style (dct_type=3)')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

    print('Finished feature extraction from ', len(featuresdf), ' files')
    return featuresdf

def divide_sets():
    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())

    # Encode the classification labels
    #le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))

    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=10)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
    x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

    num_labels = yy.shape[1]
    filter_size = 2

    return x_train, x_test, y_train, y_test, num_labels

def create_model():
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(num_labels, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # Display model architecture summary
    model.summary()

    # Calculate pre-training accuracy
    score = model.evaluate(x_test, y_test, verbose=1)
    accuracy = 100 * score[1]

    print("Pre-training accuracy: %.4f%%" % accuracy)
    return model

def train_model(model):
    # num_epochs = 40, num_batch_size = 50
    num_epochs = input("Select number of epochs: ")
    num_batch_size = input("Select batch size: ")


    # checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn.hdf5',verbose=1, save_best_only=True)
    start = datetime.now()

    #model.fit  # (x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)
    history = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs,
                        validation_data=(x_test, y_test), verbose=1)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    model.save('saved_model/myNetwork.h5')

    # Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: ", score[1])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: ", score[1])

    return model

def save_my_model(model):
    path_to_model = input("Insert model name or path: ")
    return model.save(path_to_model)

def load_my_model():
    path_to_model = input("Insert path to model: ")
    model = load_model(path_to_model)
    return model

def delete_my_model(model):
    del model
    return None

def prediction_file():
    a = input("Insert file path: ")
    if os.path.isfile(a):
        print_prediction(a)
    else: print("Incorrect file path")
    return None



def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccs


def extract_feature(file_name):
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None, None

    return np.array([mfccsscaled])


def print_prediction(file_name):
    num_rows = 40
    num_columns = 431
    num_channels = 1

    prediction_feature = extract_features(file_name)
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector)
    print("The predicted class is:", predicted_class[0], '\n')

    predicted_proba_vector = model.predict_proba(prediction_feature)
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)):
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )

# Load imports

import IPython.display as ipd
import librosa
import os
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile as wav
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


#3
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
from keras.callbacks import ModelCheckpoint
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from keras.models import model_from_json

from wavfilehelper import WavFileHelper



filename = 'sounds/sounds/pasaz_rondo_2_04(3).wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)



metadata = pd.read_csv('sounds/sounds/labels.csv')
print(metadata.head())

print(metadata.shape)

#wavfilehelper = WavFileHelper()

audiodata = []
features = []

filenames = metadata['fileName'].values
filelabels = metadata['fileLabel'].values

for index, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath("..\\untitled5\\sounds\\sounds"), filenames[index])
    class_label = filelabels[index]
    data = extract_features(file_name)

    features.append([data, class_label])


# Convert into a Panda dataframe
plt.figure(figsize=(10, 4))
librosa.display.specshow(data, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()

audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
m_slaney = librosa.feature.mfcc(y=audio, sr=sample_rate, dct_type=2)
m_htk = librosa.feature.mfcc(y=audio, sr=sample_rate, dct_type=3)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
librosa.display.specshow(m_slaney, x_axis='time')
plt.title('RASTAMAT / Auditory toolbox (dct_type=2)')
plt.colorbar()
plt.subplot(2, 1, 2)
librosa.display.specshow(m_htk, x_axis='time')
plt.title('HTK-style (dct_type=3)')
plt.colorbar()
plt.tight_layout()
plt.show()

featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')



# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 10)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


#4 poradnika !!
num_rows = 40
num_columns = 431
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]
filter_size = 2

exit = False;
while exit == False:
    print("Wroclaw places, sound recognition.")
    print("Menu:")
    print("1. Create model")
    print("2. Train model")
    print("3. Save model to file")
    print("4. Delete current model")
    print("5. Predict")
    print("6. Load model")
    print("7. Evaluate score")
    print("9, Exit")
    val = input("Choose your option: ")
    choice = int(val)
    if(choice == 1):
        model = create_model()
    elif(choice == 2):
        model = train_model(model)
    elif(choice == 3):
        save_my_model(model)
    elif(choice == 4):
        delete_my_model(model)
    elif(choice == 5):
        prediction_file()
    elif(choice == 6):
        model = load_my_model()
    elif(choice == 7):
        score = model.evaluate(x_train, y_train, verbose=0)
        print("Training Accuracy: ", score[1])
        score = model.evaluate(x_test, y_test, verbose=0)
        print("Testing Accuracy: ", score[1])
    elif(choice == 9):
        exit = True
    else:
        print("Invalid option. Try once again.")
#model = create_model()
#model = train_model(model)
#save_model(model)
#prediction_file('test/test_tramwaj6.wav')


# Construct model
"""
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# Display model architecture summary
model.summary()

# Calculate pre-training accuracy
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)



num_epochs = 40
num_batch_size = 50

#checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn.hdf5',verbose=1, save_best_only=True)
start = datetime.now()

model.fit#(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)
history = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

duration = datetime.now() - start
print("Training completed in time: ", duration)

model.save('saved_model/myNetwork.h5')

# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])



model = load_model('saved_model/myNetwork.h5')

num_rows = 40
num_columns = 431
num_channels = 1

testFile = 'sounds/sounds/tramwaj_2_04(19).wav'
print_prediction(testFile)

testFile = 'sounds/sounds/pasaz_rondo_2_04(3).wav'
print_prediction(testFile)

testFile = 'sounds/sounds/tramwaj_4_04(1).wav'
print_prediction(testFile)


testFile = 'test/test1_tramwaj.wav'
print_prediction(testFile)

testFile = 'test/test2_pks.wav'
print_prediction(testFile)

testFile = 'test/test3_pasaz.wav'
print_prediction(testFile)
"""


#while True:
    #a = input("Podaj plik: ")
    #print_prediction(a)
