from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import decode_predictions
import glob
from PIL import Image
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

sOutImage="C:\\Users\\omnia\\OneDrive - Georgia State University\\Spring2019\\graph mining\\Project\datasets\\ImageDatasets\\3K dataset\\3.kittens\\"
sFlickrText = "C:\\Users\\omnia\\PycharmProjects\\ImageCaptioning\\data_flickr_text\\"

from PIL import Image
import glob
img = glob.glob(sOutImage+'*.jpg')
def split_data(l):
    temp = []
    for i in img:
        if i[len(sOutImage):] in l:
            temp.append(i)
    return temp


# extract features from each photo in the directory
def extract_features(filename):
     # load the model
     model = VGG16()
     # load the photo
     image = load_img(filename, target_size=(224, 224))
     # convert the image pixels to a numpy array
     image = img_to_array(image)
     # reshape data for the model
     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
     # prepare the image for the VGG model
     image = preprocess_input(image)
     # get features
     yhat = model.predict(image)
     # convert the probabilities to class labels
     label = decode_predictions(yhat)
     # retrieve the most likely result, e.g. highest probability
     label = label[0][0]
     print((label[1]))

     return label[1]





filename = sFlickrText + 'DogTemp.txt'
test_images = set(open(filename, 'r').read().strip().split('\n'))

test_img = split_data(test_images)
imgList = list()
for i in range(558):
        photo = extract_features(test_img[i+551])
        #image1=Image.open(test_img[i])
        #image1.show()
