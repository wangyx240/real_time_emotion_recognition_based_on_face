from keras.models import load_model
import numpy as np
import os
import pickle
import keras 

folder_path = os.getcwd()
train_path = "dataset\\resolution_64x64x1_train.pickle"
test_path = "dataset\\resolution_64x64x1_test.pickle"
emotions_count = 8

def split_labels(data):
    x = []
    y = []
    for item in data:
        for i in range(0, len(item["images"]), 1):
            x.append(np.expand_dims(item["images"][i],axis=3))
            y.append(item["emotion"])
    return np.asarray(x), np.asarray(y)

if __name__ == '__main__':

    # load data
    with open(os.path.join(folder_path, train_path), 'rb') as handle:
        train_data = pickle.load(handle)
    
    with open(os.path.join(folder_path, test_path), 'rb') as handle:
        test_data = pickle.load(handle)
    
    # split labels (emotion index and frames)
    x_test, y_test = split_labels(test_data)
    x_train, y_train = split_labels(train_data)

    # load/create model
    model_path = "models\\base_emotion_classification_model.hdf5"
    model = load_model(model_path, compile=False)
    
    lr = 0.005
    _epochs = 1
    _batch_size = 8
    _decay = 0.01
    _momentum = 0.9
    sgd = keras.optimizers.SGD(lr=lr, momentum=_momentum, nesterov=True, decay=_decay)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    # train
    model.fit(x_train,
            y_train,
            epochs=_epochs, 
            validation_data=(x_test, y_test),
            batch_size=_batch_size) 

    # save
    index = 0
    name = "models\\emotion_classificator_{}.hdf5".format(index)
    name = os.path.join(folder_path, name)
    
    model.save(name)
    print("Model was saved to {}".format(name))
