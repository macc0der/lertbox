import keras
from keras.models import load_model
import numpy as np
import pandas as pd
import os
import librosa
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

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

def predict_class(model, file_name):
    num_rows = 40
    num_columns = 174
    num_channels = 1
    prediction_feature = extract_features(file_name) 
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", predicted_vector[0], '\n')
    return predicted_vector[0]



max_pad_len = 174


le = LabelEncoder()
y = ['air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music']
yy = to_categorical(le.fit_transform(y))
print(le.classes_)

new_model = load_model('saved_models/my_model.hdf5')
input_file = '../UrbanSound Dataset sample/audio/102853-8-0-0.wav'
predict_class(new_model, input_file) 
