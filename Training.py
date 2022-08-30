from os import listdir
from numpy import argmax
from numpy import array
from pickle import dump
from pickle import load
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.models import load_model
from keras.utils import to_categorical
from keras import utils as np_utils
from keras.utils import plot_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu
from keras.layers.merge import concatenate
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from pandas import DataFrame
from AuxilaryFunctions import *
from PreprocessImage import *
from PreprocessText import *
from TheModel import *
import matplotlib.pyplot as plt
from matplotlib import pyplot
from keras.callbacks import History

# directory for text
sFlickrText = "C:\\Users\\omnia\\PycharmProjects\\ImageCaptioning\\data_flickr_text\\"
sFlickrImage = "C:\\Users\\omnia\\PycharmProjects\\ImageCaptioning\\data_flickr_image\\"


if __name__ == '__main__':
    #i = 0

    ################################################## The Training Phase ############################################
    # load dev set
    filename = sFlickrText + 'Flickr_8k.trainImages.txt'
    train = load_set(filename)
    print('Dataset: %d' % len(train))
    # descriptions
    disFile = sFlickrText + 'descriptions.txt'
    train_descriptions = load_clean_descriptions(disFile, train)
    print('Descriptions: train=%d' % len(train_descriptions))
    # photo features
    featureFile = sFlickrText + 'features.pkl'
    train_features = load_photo_features(featureFile, train)
    print('Photos: train=%d' % len(train_features))
    # prepare tokenizer
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    # determine the maximum sequence length
    max_length = max_length(train_descriptions)
    print('Description Length: %d' % max_length)
    # prepare sequences
    X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features,vocab_size)
    print('Training Sequences Created') # it takes 15 min

########################################## Validation Dataset  #########################################

    # load validation set
    filename = sFlickrText + 'Flickr_8k.devImages.txt'
    test = load_set(filename)
    print('Dataset: %d' % len(test))
    # descriptions
    test_descriptions = load_clean_descriptions(disFile, test)
    print('Descriptions: test=%d' % len(test_descriptions))
    # photo features
    test_features = load_photo_features(featureFile, test)
    print('Photos: test=%d' % len(test_features))
    # prepare sequences
    X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features,vocab_size)
    print('Testing Sequences Created')
    # fit model

    # define the model
    model = define_model(vocab_size, max_length)
    # define checkpoint callback
    filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # fit model

    history=model.fit([X1train, X2train], ytrain, epochs=6, batch_size=2000, verbose=1, callbacks=[checkpoint],
              validation_data=([X1test, X2test], ytest))



    ################################################## Plot the Performance of the model #####################################
    # list all data in history
    print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


