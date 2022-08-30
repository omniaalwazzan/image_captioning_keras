from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import decode_predictions

sDogText = "C:\\Users\\omnia\\PycharmProjects\\ImageCaptioning\\data_flickr_text\\"
sOutImage="C:\\Users\\omnia\\OneDrive - Georgia State University\\Spring2019\\graph mining\\Project\datasets\\ImageDatasets\\3K dataset\\1.dog"
#sOutImage = "C:\\Users\omnia\\OneDrive - Georgia State University\\Spring2019\\graph mining\\Project\\instagram-scraper-master\\weeds\\"
'''
img = glob.glob(sOutImage+'*.jpg')
def split_data(l):
    temp = []
    for i in img:
        if i[len(sOutImage):] in l:
            temp.append(i)
    return temp
'''

# load the model
model = VGG16()
# load an image from Weeds file
II = sOutImage + '51786083_132302054491624_4466329273384986956_n.jpg'
image = load_img(II, target_size=(224, 224))
print(image)
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model.predict(image)
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
print((label[1]))
# print the classification
#print('%s (%.2f%%)' % (label[1], label[2]*100))
import os
import glob
from PIL import Image
sOutImage = "C:\\Users\omnia\\OneDrive - Georgia State University\\Spring2019\\graph mining\\Project\\instagram-scraper-master\\weeds\\"
import cv2

filenames = glob.glob(sOutImage+'*.jpg')
filenames.sort()
images = [cv2.imread(img) for img in filenames]

for img in images:
    print(img[1])


#"C:\\Users\omnia\OneDrive - Georgia State University\Spring2019\graph mining\Project\datasets\ImageDatasets\3K dataset\Generated TXT features"

#DogTemp = "C:\\Users\omnia\\OneDrive - Georgia State University\\Spring2019\\graph mining\\Project\\mageDatasets\\3K dataset\\Generated TXT features\\DogTemp.txt"


with open('C:\DogTemp.txt', 'r') as f:
    x = f.readlines()

f = open('C:\DogTemp.txt', 'r')
x = f.readlines()
for line in f:
    print(x[line])
    f.close()
list_of_lists=[];
with open('C:\DogTemp.txt', 'r') as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(',')]
        # in alternative, if you need to use the file content as numbers
        # inner_list = [int(elt.strip()) for elt in line.split(',')]
        list_of_lists.append(inner_list)

for i in range(0, len(list_of_lists)):
    print(list_of_lists[i])

# # extract features from each photo in the directory
 def extract_features(filename):
     # load the model
     model = VGG16()
     # re-structure the model
     model.layers.pop()
     model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
     # load the photo
     image = load_img(filename, target_size=(224, 224))
     # convert the image pixels to a numpy array
     image = img_to_array(image)
     # reshape data for the model
     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
     # prepare the image for the VGG model
     image = preprocess_input(image)
     # get features
     feature = model.predict(image, verbose=0)

     return feature


import glob
img = glob.glob(sOutImage+'*.jpg')
def split_data(l):
    temp = []
    for i in img:
        if i[len(sOutImage):] in l:
            temp.append(i)
    return temp






list2.append(change)


# # extract features from each photo in the directory
# def extract_features(filename):
#     # load the model
#     model = VGG16()
#     # re-structure the model
#     model.layers.pop()
#     model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
#     # load the photo
#     image = load_img(filename, target_size=(224, 224))
#     # convert the image pixels to a numpy array
#     image = img_to_array(image)
#     # reshape data for the model
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     # prepare the image for the VGG model
#     image = preprocess_input(image)
#     # get features
#     feature = model.predict(image, verbose=0)
#
#     return feature




# extract features from each photo in the directory
def extract_features(directory):
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # summarize
    print(model.summary())
    # extract features from each photo
    features = dict()
    for name in listdir(directory):
        # load an image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
        print('>%s' % name)
    return features


# extract features from all images
#directory = sOutImage
#features = extract_features(directory)
#print('Extracted Features: %d' % len(features))
# save to file
#dump(features, open('featuresGM.pkl', 'wb'))

# import pickle
#
# file = open('featuresGM.pkl', 'rb')
#
# image = pickle.load(file)
# print(image)
# file.close()

#OutImage = sOutImage + '18949970_1440064799366273_1487556434001395712_n.jpg'
#photo = extract_features(OutImage)
#print(photo)


