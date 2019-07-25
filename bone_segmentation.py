# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:55:54 2019

@author: Angelo Antonio Manzatto

Reference: https://www.ircad.fr/research/3d-ircadb-01/
3D IRCAD - Hôpitaux Universitaires
"""

##################################################################################
# Libraries
##################################################################################  
import os
import requests
import io
import random
import glob
import zipfile
import shutil 
import pydicom

import numpy as np

import matplotlib.pyplot as plt

import cv2

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Model

from keras.layers import Input,Activation
from keras.layers import Conv2D,  MaxPooling2D, BatchNormalization, Conv2DTranspose, concatenate


from keras.callbacks import ModelCheckpoint, CSVLogger

import keras.backend as K

from keras.optimizers import Adam

##################################################################################
# Download Dataset
##################################################################################  
def process_ircad_database(database_url, inputs_folder, bone_masks_folder, overwrite=False):
    
    # Database name will be assumed as the name on the last "/" before the ".zip"
    database_name = database_url.split('/')[-1].split(".zip")[0]
    
    print(50*'-')
    print("Processing: {0}".format(database_name))
    print(50*'-')
    
    # Download the dataset if doesn't exist and extract to appropriate folder
    database_folder = os.path.join(dataset_folder,database_name)
    
    if not os.path.exists(database_folder):
        os.makedirs(database_folder)
    
    # Request files
    r = requests.get(database_url)
    
    print("Request status code: {0}".format(r.status_code))
    
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    r.close()

    print("Database downloaded")
    print(50*'-')
    
    # Extract data to folder
    z.extractall(dataset_folder)
    z.close()
    
    print("Files extracted")
    print(50*'-')
    
    # Unzip Laled dicom
    label_dicom_zip_file = os.path.join(database_folder,'LABELLED_DICOM.zip')
    label_dicom_zip = zipfile.ZipFile(label_dicom_zip_file)
    label_dicom_zip.extractall(database_folder)
    label_dicom_zip.close()
    
    # Unzip masks dicom
    mask_dicom_zip_file = os.path.join(database_folder,'MASKS_DICOM.zip')
    mask_dicom_zip = zipfile.ZipFile(mask_dicom_zip_file)
    mask_dicom_zip.extractall(database_folder)
    mask_dicom_zip.close()
    
    print("Label and Masks extracted")
    print(50*'-')
    
    # Folders form input and bone masks dicom
    label_dicom_folder = os.path.join(database_folder,'LABELLED_DICOM') 
    bone_mask_dicom_folder = os.path.join(database_folder,'MASKS_DICOM','bone') 
    
    # Select all input and mask dicom files
    label_dicom_paths = glob.glob(os.path.join(label_dicom_folder,'*')) 
    bone_mask_dicom_paths = glob.glob(os.path.join(bone_mask_dicom_folder,'*')) 
    
    n_label = len(label_dicom_paths)
    n_bone_masks = len(bone_mask_dicom_paths) 
    print("Label files: {0}".format(n_label))
    print("Bone mask files: {0}".format(n_bone_masks))
    print(50*'-')
    # Check if we have files and both input and mask folder have the same number of files
    assert(len(label_dicom_paths) == len(bone_mask_dicom_paths) and len(label_dicom_paths) > 0 and len(bone_mask_dicom_paths) > 0)

    # Move files to appropriate folder

    current_index = 1
    
    # Update the current_index with the last file index on the folder
    if(overwrite == False):
        
        input_dicom_paths = glob.glob(os.path.join(inputs_folder,'*.dicom'))
        
        if(len(input_dicom_paths) == 0):
            current_index = 1
        else:
            # Grab the file names and extract just the number so input00012.dicom -> 12
            indices = [int(x.split("\\")[-1][5:10])  for x in input_dicom_paths]
            indices.sort()
            last_index = indices[-1]
            
            current_index = last_index + 1
    
    print("Moving files starting with index: {0}".format(current_index))
    print(50*'-')     
    
    for label_dicom_path, bone_mask_dicom_path in zip(label_dicom_paths, bone_mask_dicom_paths):
    
        assert(label_dicom_path.split("\\")[-1] == bone_mask_dicom_path.split("\\")[-1])
        
        new_input_dicom_path = os.path.join(inputs_folder,'input' + str(current_index).zfill(5) + ".dicom")
        new_bone_mask_dicom_path = os.path.join(bone_masks_folder,'mask' + str(current_index).zfill(5) + ".dicom")
        
        shutil.copy(label_dicom_path, new_input_dicom_path)
        shutil.copy(bone_mask_dicom_path, new_bone_mask_dicom_path)
        
        current_index +=1
    
    print("Moving process complete")
    print(50*'-')    
    
    # Delete temporaty database folder after we exctracted files we wantes
    shutil.rmtree(database_folder)
    
    print("Database folder excluded")
    print(50*'-')  
            

dataset_folder = 'dataset'

inputs_folder = os.path.join(dataset_folder,'inputs')
bone_masks_folder = os.path.join(dataset_folder,'bone_masks')

# Create Dataset folder 
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)
    
# Create Input DICOM folder 
if not os.path.exists(inputs_folder):
    os.makedirs(inputs_folder)
    
# Create Bone Masks folder
if not os.path.exists(bone_masks_folder):
    os.makedirs(bone_masks_folder)
    
# Uncomment to process again <!!!!!! REMOVE FOR FINAL PUBLISHING>
database_urls = ['''
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.1.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.2.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.3.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.4.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.5.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.6.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.7.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.8.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.9.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.10.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.11.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.12.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.13.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.14.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.15.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.16.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.17.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.18.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.19.zip',
                 'https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.20.zip',
                 '''
                 ]

# The request library hangs sometimes after processing the last request. If that happens you have to start the process
# again from the dataset that you have stoped
for database_url in database_urls:
    process_ircad_database(database_url,inputs_folder,bone_masks_folder)

############################################################################################
# Load data
############################################################################################
    
# DICOM image files
input_files = glob.glob(os.path.join(inputs_folder,'*.dicom'))  
bone_mask_files = glob.glob(os.path.join(bone_masks_folder,'*.dicom'))   

# Sort names to avoid connecting wrong numbered files
input_files.sort()
bone_mask_files.sort()

assert(len(input_files) == len(bone_mask_files))

# Our dataset is created using the tuple (input image file, ground truth image file)
data = []
for input_image, bone_mask_image in zip(input_files, bone_mask_files):
    
    data.append((input_image,bone_mask_image))

# Plot some samples from training set
n_samples = 5

for i in range(n_samples):
    
    # define the size of images
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.set_figwidth(10)
    
    # randomly select a sample
    idx = np.random.randint(0, len(data))
    input_image_path, bone_mask_image_path = data[idx]
    
    input_image = pydicom.read_file(input_image_path).pixel_array
    bone_mask_image = pydicom.read_file(bone_mask_image_path).pixel_array
    
    ax1.imshow(input_image, cmap = 'gray')
    ax1.set_title('Input Image # {0}'.format(idx))
    
    ax2.imshow(bone_mask_image, cmap = 'gray')
    ax2.set_title('Bone Mask Image # {0}'.format(idx))
    
############################################################################################
# Data Augmentation 
############################################################################################
    
#############################
# Resize Image
#############################
class Resize(object):
    
    def __init__(self, input_size, output_size):
        
        assert isinstance(input_size,(int, tuple))
        assert isinstance(output_size,(int, tuple))
        
        self.input_size = input_size
        self.output_size = output_size    
        
    def __call__(self, input_image, mask_image):     
           
        h_in, w_in = input_image.shape[:2]
        h_out, w_out = mask_image.shape[:2]
        
        if isinstance(self.input_size, int):
            if h_in > w_in:
                new_h_in, new_w_in = self.input_size * h_in / w_in, self.input_size
            else:
                new_h_in, new_w_in = self.input_size, self.input_size * w_in / h_in
        else:
            new_h_in, new_w_in = self.input_size
        
        if isinstance(self.output_size, int):
            if h_out > w_out:
                new_h_out, new_w_out = self.output_size * h_out / w_out, self.output_size
            else:
                new_h_out, new_w_out = self.output_size, self.output_size * w_out / h_out
        else:
            new_h_out, new_w_out = self.output_size

        new_h_in, new_w_in = int(new_h_in), int(new_w_in)
        new_h_out, new_w_out = int(new_h_out), int(new_w_out)

        input_image = cv2.resize(input_image, (new_w_in, new_h_in))
        mask_image = cv2.resize(mask_image, (new_w_out, new_h_out)) 
        
        return input_image,  mask_image

#############################
# Translate Image
#############################    
class RandomTranslation(object):

    def __init__(self, ratio = 0.4, background_color = (0) , prob=0.5):
        
        self.background_color  = background_color
        self.ratio = ratio
        self.prob  = prob
        
    def __call__(self, input_image, mask_image):
                
        if random.uniform(0, 1) <= self.prob:
            
            img_h, img_w = input_image.shape
            
            x = int(np.random.uniform(-self.ratio,self.ratio) * img_w)
            y = int(np.random.uniform(-self.ratio,self.ratio) * img_h)

            M = np.float32([[1, 0, x],
                            [0, 1, y]])
                
            input_image_translated = cv2.warpAffine(input_image,M,(img_w,img_h), borderValue=self.background_color)
            imask_image_translated = cv2.warpAffine(mask_image,M,(img_w,img_h), borderValue=self.background_color)
            
            return input_image_translated , imask_image_translated
           
        return input_image, mask_image

#############################
# Scale Image
#############################  
class RandomScale(object):

    def __init__(self, lower = 0.4, upper = 1.4, background_color = (0) , prob=0.5):
        
        self.background_color = background_color
        self.lower = lower
        self.upper = upper
        self.prob  = prob  
        
    def __call__(self, input_image, mask_image):
                
        if random.uniform(0, 1) <= self.prob:
            
            input_img_h, input_img_w = input_image.shape
            mask_img_h, mask_img_w = mask_image.shape
            
             # Create canvas with random ration between lower and upper
            ratio = random.uniform(self.lower,self.upper)
            
            scale_x = ratio
            scale_y = ratio
            
            # Scale the image
            scaled_input_image = cv2.resize(input_image.astype('float32'),(0,0),fx=scale_x,fy=scale_y)
            scaled_mask_image = cv2.resize(mask_image.astype('float32'),(0,0),fx=scale_x,fy=scale_y)
            
            top = 0
            left = 0
            
            if ratio < 1:
                    
                # Input image
                background = np.zeros((input_img_h, input_img_w), dtype = np.uint8)
                
                background[:,:] = self.background_color 
            
                y_lim = int(min(scale_x,1)*input_img_h)
                x_lim = int(min(scale_y,1)*input_img_w)
                
                top  = (input_img_h - y_lim) // 2
                left = (input_img_w - x_lim) // 2

                background[top:y_lim+top,left:x_lim+left] = scaled_input_image[:y_lim,:x_lim]
                
                scaled_input_image = background
                
                # Mask image                
                background = np.zeros((mask_img_h, mask_img_w), dtype = np.uint8)
                
                background[:,:] = self.background_color 
                
                y_lim = int(min(scale_x,1)*mask_img_h)
                x_lim = int(min(scale_y,1)*mask_img_w)
                
                top  = (mask_img_h - y_lim) // 2
                left = (mask_img_w - x_lim) // 2
                
                background[top:y_lim+top,left:x_lim+left] = scaled_mask_image[:y_lim,:x_lim]
                
                scaled_mask_image = background
                    
            else:
                
                top  = (scaled_input_image.shape[0] -  input_img_h) // 2
                left = (scaled_input_image.shape[1] -  input_img_w) // 2
                
                scaled_input_image = scaled_input_image[top:input_img_h+top,left:input_img_w+left]
                
                top  = (scaled_mask_image.shape[0] -  mask_img_h) // 2
                left = (scaled_mask_image.shape[1] -  mask_img_w) // 2
                
                scaled_mask_image = scaled_mask_image[top:mask_img_h+top,left:mask_img_w+left]
                         
            return scaled_input_image, scaled_mask_image

        return input_image, mask_image
    
#############################
# Flip image
#############################    
class RandomFlip(object):
 
    def __init__(self, prob=0.5):

        self.prob = prob
        
    def __call__(self, input_image, mask_image):
        
        if random.uniform(0, 1) <= self.prob:
        
            # Get image shape
            h, w = input_image.shape[:2]
            
            # Flip image
            input_image = input_image[:, ::-1]
            
            # Get image shape
            h, w = mask_image.shape[:2]
            
            # Flip image
            mask_image = mask_image[:, ::-1]
            
            return input_image, mask_image
        
        return input_image, mask_image
    
#############################    
# Gaussian Noise
#############################    
class GaussianNoise(object):
    
    def __init__(self, prob=0.5):

        self.prob = prob
    
    def __call__(self, input_image, mask_image):
        
        if random.uniform(0, 1) <= self.prob:
        
             # Get image shape
            h, w = input_image.shape[:2]
            
            mean = 0
            var = 0.1
            sigma = var**0.5
            
            gauss = np.random.normal(mean,sigma,(w,h))
            gauss = gauss.reshape(w,h)
            
            noise_input_image = input_image + gauss
            
            return noise_input_image, mask_image
        
        return input_image, mask_image


#############################
# Normalize
#############################
class Normalize(object):
    
    """Normalize the color range to [0,1].""" 
       
    def __call__(self, input_image, mask_image):
               
        image_copy = input_image.copy().astype(np.float)
        label_copy = mask_image.copy().astype(np.float)
        
        M = np.float(np.max(image))
        if M != 0:
            image_copy  *= 1./M
            
        M = np.float(np.max(label))
        if M != 0:
            label_copy  *= 1./M
        

        return ( image_copy, label_copy)   

############################################################################################
# Test Data Augmentation 
############################################################################################
def plot_transformation(transformation, n_samples = 3):

    for i in range(n_samples):
    
        # define the size of images
        f, (ax1, ax2) = plt.subplots(1, 2)
        f.set_figwidth(10)

        # randomly select a sample
        idx = np.random.randint(0, len(data))
        input_image_path, bone_mask_image_path = data[idx]
    
        input_image = pydicom.read_file(input_image_path).pixel_array
        bone_mask_image = pydicom.read_file(bone_mask_image_path).pixel_array

        new_input_image, new_gt_image= transformation(input_image, bone_mask_image)

        ax1.imshow(new_input_image , cmap = 'gray')
        ax1.set_title('Original')


        ax2.imshow(new_gt_image , cmap = 'gray')
        ax2.set_title(type(transformation).__name__)

        plt.show()
        
##########################
# Resize Test
##########################
resize = Resize((512, 512),(512, 512))
plot_transformation(resize)

##########################
# Random Translation
##########################
translation = RandomTranslation(ratio = 0.2, prob=1.0)
plot_transformation(translation)

##########################
# Random Translation
##########################
scale = RandomScale(prob=1.0)
plot_transformation(scale)

##########################
# Random Flip Test
##########################
flip = RandomFlip()
plot_transformation(flip)

##########################
# Random Flip Test
##########################
noise = GaussianNoise(prob=1.0)
plot_transformation(noise)

############################################################################################
# Create Dataset
############################################################################################
    
# Create X, y tuple from image_path, key_pts tuple
def createXy(data, transformations = None):
    
    input_image_path, bone_mask_image_path = data
    
    input_image = pydicom.read_file(input_image_path).pixel_array 
    bone_mask_image = pydicom.read_file(bone_mask_image_path).pixel_array 
    
    # Apply transformations for the tuple (image, labels, boxes)
    if transformations:
        for t in transformations:
            input_image, bone_mask_image = t(input_image, bone_mask_image)
            
    input_image = np.expand_dims(input_image, axis = -1)     
    bone_mask_image = np.expand_dims(bone_mask_image, axis = -1)            
            
    return input_image, bone_mask_image

# Generator for using with model
def generator(data, transformations = None, batch_size = 4, shuffle_data= True):
    
    n_samples = len(data)
    
    # Loop forever for the generator
    while 1:
        
        if shuffle_data:
            data = shuffle(data)
        
        for offset in range(0, n_samples, batch_size): 
            
            batch_samples = data[offset:offset + batch_size]
            
            X = []
            y = []
            
            for sample_data in batch_samples:
                
                image, target = createXy(sample_data, transformations)

                X.append(image)
                y.append(target)
                
            X = np.asarray(X).astype('float32')
            y = np.asarray(y).astype('float32')
            
            yield (shuffle(X, y))
            
############################################################################################
# Unet Model
############################################################################################
def unet_encoder(inputs,filters, block_id):
    
    x = Conv2D(filters, kernel_size=(3,3), padding='same' , kernel_initializer='he_normal', name='block_' + str(block_id) + '_unet_encoder_conv2d_1')(inputs)
    x = BatchNormalization(name='block_' + str(block_id) + '_unet_encoder_conv_batch_1')(x)
    x = Activation('relu', name='block_' + str(block_id) + '_unet_encoder_relu_1')(x)
    
    x = Conv2D(filters, kernel_size=(3,3), padding='same' , kernel_initializer='he_normal', name='block_' + str(block_id) + '_unet_encoder_conv2d_2')(x)
    x = BatchNormalization(name='block_' + str(block_id) + '_unet_encoder_conv_batch_2')(x)
    x = Activation('relu', name='block_' + str(block_id) + '_unet_encoder_relu_2')(x)
    
    return x
                
def unet_encoder_pool(inputs,filters, block_id):
    
    x = unet_encoder(inputs, filters, block_id)
    x = MaxPooling2D(pool_size=(2,2), name='block_' + str(block_id) + '_unet_pooling')(x)
    
    return x

def unet_decoder(inputs_a, inputs_b, filters, block_id):
    
    x = Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2), padding='same')(inputs_a)
    
    x = concatenate([x, inputs_b],axis=3)
    
    x = Conv2D(filters, kernel_size=(3,3), padding='same' , kernel_initializer='he_normal', name='block_' + str(block_id) + '_unet_decoder_conv2d_1')(x)
    x = BatchNormalization(name='block_' + str(block_id) + '_unet_decoder_conv_batch_1')(x)
    x = Activation('relu', name='block_' + str(block_id) + '_unet_decoder_relu_1')(x)
    
    x = Conv2D(filters, kernel_size=(3,3), padding='same' , kernel_initializer='he_normal', name='block_' + str(block_id) + '_unet_decoder_conv2d_2')(x)
    x = BatchNormalization(name='block_' + str(block_id) + '_unet_decoder_conv_batch_2')(x)
    x = Activation('relu', name='block_' + str(block_id) + '_unet_decoder_relu_2')(x)
    
    return x
              
def UNet(input_shape=(512,512,1)):
    
    Image = Input(shape=input_shape)
    
    encoder = unet_encoder_pool(Image,   filters=32,  block_id=0)
    encoder = unet_encoder_pool(encoder, filters=64,  block_id=1)
    encoder = unet_encoder_pool(encoder, filters=128, block_id=2)
    encoder = unet_encoder_pool(encoder, filters=256, block_id=3)
    encoder = unet_encoder(encoder, filters=512, block_id=4)
    
    encoder_model = Model(inputs=Image,outputs=encoder)

    tensor1 = encoder_model.get_layer('block_3_unet_encoder_relu_2').output
    tensor2 = encoder_model.get_layer('block_2_unet_encoder_relu_2').output
    tensor3 = encoder_model.get_layer('block_1_unet_encoder_relu_2').output
    tensor4 = encoder_model.get_layer('block_0_unet_encoder_relu_2').output
    
    decoder = unet_decoder(encoder, tensor1, filters=256, block_id = 4)
    decoder = unet_decoder(decoder, tensor2, filters=128, block_id = 5)
    decoder = unet_decoder(decoder, tensor3, filters=64, block_id = 6)
    decoder = unet_decoder(decoder, tensor4, filters=32, block_id = 7)
    
    decoder = Conv2D(1,kernel_size=(1,1),strides=(1,1), activation='sigmoid')(decoder)
    
    model = Model(inputs=Image, outputs=decoder)
    
    return model

############################################################################################
# Unet Loss
############################################################################################
class UNetLoss():
    
    def __init__(self, smooth = 1):
        
        self.smooth = smooth
    
    def calculate_loss(self, y_true, y_pred):
        
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        
        intersection = K.sum(y_true_f * y_pred_f)
        
        return (2. * intersection + self.smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth)

############################################################################################
# Training pipeline
############################################################################################

# Data augmentation 
train_transformations = [
        Resize((512,512),(512,512))
        ]

test_transformations = [
        Resize((512,512),(512,512))
        ]

valid_transformations = [
        Resize((512,512),(512,512))
        ]

# Hyperparameters
epochs = 200
batch_size = 1 # Change this value if you have more GPU Power
learning_rate = 0.001
weight_decay = 5e-4
momentum = .9

train_valid_data, test_data = train_test_split(data, test_size=0.20, random_state=42)
train_data, valid_data = train_test_split(train_valid_data, test_size=0.25, random_state=42)

print('Size of train data: {0}'.format(len(train_data)))
print('Size of test data: {0}'.format(len(test_data)))
print('Size of valid data: {0}'.format(len(valid_data)))

train_generator = generator(train_data, train_transformations, batch_size)
test_generator = generator(test_data, test_transformations, batch_size)

# callbacks
model_path = 'saved_models'

model = UNet()
model.summary()

# Create loss function
loss = UNetLoss(smooth = 1)

# Create Optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Compile model for training
model.compile(optimizer, loss = loss.calculate_loss)

# File were the best model will be saved during checkpoint     
model_file = os.path.join(model_path,'bone_segmentation-{val_loss:.4f}.h5')

# Check point for saving the best model
check_pointer = ModelCheckpoint(model_file, monitor='val_loss', mode='min',verbose=1, save_best_only=True)

# Logger to store loss on a csv file
csv_logger = CSVLogger(filename='bone_segmentation.csv',separator=',', append=True)

history = model.fit_generator(train_generator,steps_per_epoch=int(len(train_data) / batch_size),
                              validation_data=test_generator,validation_steps=int(len(test_data) / batch_size),
                              epochs=epochs, verbose=1, callbacks=[check_pointer,csv_logger],workers=1)