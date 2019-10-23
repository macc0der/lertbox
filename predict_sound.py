from keras.models import load_model

def predict_class(model, file_name):
    prediction_feature = extract_features(file_name) 
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    #print("The predicted class is:", predicted_vector[0], '\n')
    return predicted_vector[0]

new_model = load_model('my_model.hdf5')
input_file = 'audio/102853-8-0-0.wav'
predict_class(new_model, input_file) 