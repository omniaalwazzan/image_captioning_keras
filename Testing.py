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
import glob
from PIL import Image
from AuxilaryFunctions import *
from PreprocessImage import *
from PreprocessText import *
from TheModel import *
from Training import *
from keras import utils as np_utils
from pickle import dump
# directory for text
sFlickrText = "C:\\Users\\omnia\\PycharmProjects\\ImageCaptioning\\data_flickr_text\\"
sFlickrImage = "C:\\Users\\omnia\\PycharmProjects\\ImageCaptioning\\data_flickr_image\\"
SavedModel= "C:\\Users\\omnia\\PycharmProjects\\ImageCaptioning\\src\\SavedModels\\"
sOutImage = "C:\\Users\\omnia\\PycharmProjects\\ImageCaptioning\\ImageFromOut\\"
img = glob.glob(sFlickrImage+'*.jpg')
def split_data(l):
    temp = []
    for i in img:
        if i[len(sFlickrImage):] in l:
            temp.append(i)
    return temp

#test_images_file = 'Flickr8k_text/Flickr_8k.testImages.txt'

if __name__ == '__main__':
    #i = 0
    # load test set
    filename = sFlickrText +'Flickr_8k.testImages.txt'
    test = load_set(filename)
    # descriptions
    disFile = sFlickrText + 'descriptions.txt'
    test_descriptions = load_clean_descriptions(disFile, test)
    # photo features
    featureFile = sFlickrText + 'features.pkl'
    test_features = load_photo_features(featureFile, test)
    #This has been created again for displaying imag during testing
    test_images = set(open(filename, 'r').read().strip().split('\n'))
    test_img = split_data(test_images)

    filename = sFlickrText + 'Flickr_8k.trainImages.txt'
    train = load_set(filename)
    train_descriptions = load_clean_descriptions(disFile, train)
    tokenizer = load(open('tokenizer.pkl', 'rb'))
    max_length = max_length(train_descriptions)

    #models are all my expirements in this project, each model has differnt BLEU score and recongnize differnt parts of the picture
    #The best model which has the lowes validation loss
    Modelname = SavedModel +'model-ep005-loss3.033-val_loss3.692.h5'
    #model = load_model(Modelname)
    #evaluate model
    #evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

    ################################################## Testing Phase #####################################
    '''for i in range(5):

        photo = extract_features(test_img[i+666]) #85 is good
        #print (test_img[i]
        image1=Image.open(test_img[i+666])
        #image1.show()
        display(image1)
        model = load_model(Modelname)
        description = generate_desc(model, tokenizer, photo, max_length)
        print(description)
    '''
    ################################################## Image from Out#####################################
    OutImage = sOutImage + 'example.jpg'
    photo = extract_features(OutImage)
    model = load_model(Modelname)

    description = generate_desc(model, tokenizer, photo, max_length)
    print(description)

