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
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
# directory for text
sFlickrText = "C:\\Users\\omnia\\PycharmProjects\\ImageCaptioning\\data_flickr_text\\"
sFlickrImage = "C:\\Users\\omnia\\PycharmProjects\\ImageCaptioning\\data_flickr_image\\"
sOutImage = "C:\\Users\\omnia\\PycharmProjects\\ImageCaptioning\\ImageFromOut\\"
SavedModel= "C:\\Users\\omnia\\PycharmProjects\\ImageCaptioning\\src\\SavedModels\\"

if __name__ == '__main__':
    #i = 0
    filename = sFlickrText +'Flickr_8k.testImages.txt'
    test = load_set(filename)
    print('Dataset: %d' % len(test))
    # descriptions
    disFile = sFlickrText + 'descriptions.txt'
    test_descriptions = load_clean_descriptions(disFile, test)
    print('Descriptions: test=%d' % len(test_descriptions))
    # photo features
    featureFile = sFlickrText + 'features.pkl'
    test_features = load_photo_features(featureFile, test)
    print('Photos: test=%d' % len(test_features))
    disFile = sFlickrText + 'descriptions.txt'
    filename = sFlickrText + 'Flickr_8k.trainImages.txt'
    train = load_set(filename)
    train_descriptions = load_clean_descriptions(disFile, train)
    max_length = max_length(train_descriptions)
    # load the model
    tokenizer = load(open('tokenizer.pkl', 'rb'))
    # pre-define the max sequence length (from training)
    max_length = 34
    #################################################################Evalute the model and get the BLEU score #############################
    filename = SavedModel +'model-ep005-loss3.033-val_loss3.692.h5'
    model = load_model(filename)
    # evaluate model
    evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
    ########################################################### Generate a discription for random image from the internet
    #load and prepare the photograph ,this photo is not from the test data ,it's from the internet ,any one can download a photo and place in
    # sOutImage file ,then see how our model performs
    OutImage = sOutImage + 'ppl.jpg'
    photo = extract_features(OutImage)
    # evaluate model
    description = generate_desc(model, tokenizer, photo, max_length)
    print(description)

