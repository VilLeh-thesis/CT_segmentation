#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 12:59:33 2025

@author: vilmalehto

A code for actually running the model.

First the root folder and number of segmentation classes is set. If needed the 
model file and files for saving the model and results are determined. This can 
be done by asking these as a input from user or just setting them in the main 
function. The decision for used method is done by changing the commenting in 
the main function.

The program asks the user if they want to train the model, predict with existing
model or close the program. 

For training the program calls modul_training function which sets the names
of the image and label files and then calls the ModuleTraining class. The class
does the training and saves the weights of the model for use. 

For predicting the program loads the model and weights and calls the predicting
function. The functions does the prediction and saves the results. 

If the user wants to close the program the input loop is closed and the 
program stops running.

For the whole program to work you have to have a root folder including a 
file for all the images. The model and pre-existing weights can be in or out
the root folder. 

The file structure:
    root -
         /
          - train_images
         /
         /
          - train_labels
         /
         /
          - validation_images
         /
         /
          - validation_labels
         /
         /
          - test_images (images for prediction)
         /
         /
          - predictions
         
         
     Model/Weights
         
         
    Checkpoint
        
"""
import cv2
import os
import torch
import nibabel as nib
import numpy as np

from torch.utils.data import DataLoader

#my classes
from data_preparation import Dataset
from train_model import ModuleTraining
from train_model import Predicting
from unet import UNet

    
    
def set_parameters(mode):
    """
    Function for setting the basic parameters for the code to run. Asks if the 
    user is training or predicting and if the training is continued from 
    previous session. Based on this information sets the 
    number of classes, initializes the resume parameter and sets the paths
    to the data folders. 

    Parameters
    ----------
    mode : string
        Either "T" for training or "P" for predicting

    Returns
    -------
    root_path : string
        Path to the root folder. 
    num_classes : int
        Number of segmented classes. 
    images : [string, string]
        A list including the names of the training and validation images folders.
    labels : [string, string]
        A list including the names of the training and validation labels folders.
    model_path : string
        Path to the file where the model will be saved. Or when predicting
        path to the file where the weights are saved. 
    resume : string
        Either "Y" (yes) for continueing previous training session, or
        "N" (no) for not continueing previous training. 
    """
    
    #ask the number of segments
    #num_classes = int(input("Give the number of segments (including background): "))
    num_classes = 7
    
    resume = "NaN"
    
    root_path = input("To run the model give the path to a root folder including the folders for images and model: ")
    
    if mode == "T":
        #train_img = input("What is the name of the folder for training images: ")
        train_img = "Training_images"
        #train_labels = input("What is the name of the folder for training labels: ")
        train_labels = "Training_labels"
        #val_img = input("What is the name of the file for validation images: ")
        val_img = "Validation_images"
        #val_labels = input("What is the name of the folder for validation labels: ")
        val_labels = "Validation_labels"
        
        images = [train_img, val_img]
        labels = [train_labels, val_labels]
        
        #model_path = input("Give the path to the folder where model will be saved: ")
        model_path = os.path.join(root_path, "Model")
        
        resume = input("Are you countinuing old training session? (Y/N) ")
        
        while resume not in ("N", "Y"):
            print("The input must be Y or N.")
            resume = input("Are you continuing old training session? (Y/N) ")
        
    if mode == "P":
        #model_path = input("Give path to the file where model weights are saved: ")
        model_path = os.path.join(root_path, "Model/unet.pth")
        
        #labels= input("Give the name of the folder where predictions will be saved: ")
        labels = "Predictions"
        
        #images = input("Give the name of the folder where the images are: ")
        images = "Test_images"
        
    return root_path, num_classes, images, labels, model_path, resume

def contrastAdjustment(image):
    """
    A function for modifying the CT images contrast levels. Each slice 
    is normalized to the range [0, 255] and processed with 
    Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to improve local contrast. The enhanced slices are combined into a volume
    with the same shape as the input image.

    Parameters
    ----------
    image : numpu.ndarray
        oxel data array extracted from the CT image using "image.get_fdata()".

    Returns
    -------
    enhanced_volume : numpy.ndarray
        3D array storing the CLAHE-enhanced version of the input volume.

    """
    clahe = cv2.createCLAHE(clipLimit = 3.5)
    
    enhanced_volume = np.zeros_like(image)
    #apply clahe slice by slice
    for i in range(image.shape[2]):
        current_slice = image[:, :, i]
        slice_norm = cv2.normalize(current_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        enhanced_volume[:, :, i] = clahe.apply(slice_norm)
    
    return enhanced_volume



def volumeAugmentation(image, label):
    """
    A function for data augmentation. The training and validation images are 
    augmented by rotating, zooming and flipping those.
    
    Parameters
    ----------
    image : numpy.ndarray 
        3D array storing the CLAHE-enhanced version of the input volume. The 
        output of the contrastAdjustment function. 
    label : numpy.ndarry
        Voxel data array extracted from the label image using "label.get_fdata()".
        Contains label values for each voxel in the image volume.
    
    Returns
    -------
    augmented_image : numpy.ndarry
        Voxel data array including the new augmented image. 
    augmented_label : numpy.ndarray
        Voxel data array including the new augmented label.
    image : numpy.ndarray
        The input image, for saing purposes. 
    label : numpy.ndarray
        The input label, for saving purposes.

    """
    #set the limits for augmentations
    flip_probability = 0.5
    zoom_range = 0.05
    rotation_range = 5
    
    #get the dimensions of image volume
    h, w, d = image.shape
    
    #set the augmentation parameters
    do_flip = np.random.rand() < flip_probability
    zoom_factor = 1 + np.random.uniform(-zoom_range, zoom_range)
    angle = np.random.uniform(-rotation_range, rotation_range)
    
    zoom = cv2.getRotationMatrix2D((w/2, h/2), 0, zoom_factor)
    rotation = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    
    #init the augmented volume
    augmented_image = np.zeros_like(image)
    augmented_label = np.zeros_like(label)
    
    #apply augmentation for all slices in the image volume
    #all will be zoomed and rotated, but the flipping will happen only in random cases
    for i in range(d):
        current_image_slice = image[:, :, i]
        current_label_slice = label[:, :, i]
        
        if do_flip:
            current_image_slice = cv2.flip(current_image_slice, 1)
            current_label_slice = cv2.flip(current_label_slice, 1)
            
        #zooming   
        current_image_slice = cv2.warpAffine(current_image_slice, zoom, (w, h), flags=cv2.INTER_LINEAR)
        current_label_slice = cv2.warpAffine(current_label_slice, zoom, (w, h), flags=cv2.INTER_LINEAR)
        
        #rotation
        current_image_slice = cv2.warpAffine(current_image_slice, rotation, (w, h), flags = cv2.INTER_NEAREST)
        current_label_slice = cv2.warpAffine(current_label_slice, rotation, (w, h), flags = cv2.INTER_NEAREST)
        
        augmented_image[:, :, i] = current_image_slice
        augmented_label[:, :, i] = current_label_slice

        
    return augmented_image, augmented_label, image, label



def pre_process(images, labels, mode):
    """
    Based on if the user is training or predicting combines and calls the
    necessary preprocessing steps for images and labels. 

    Parameters
    ----------
    images : string
        DThe path to the processed images. 
    labels : string
        The path to the processesd labels. 
    mode : string 
        "T" for training or "P" for predicting

    Returns
    -------
    output_path : [string, string]
        A list f output paths, one to the folder containing the processed 
        images and to the folder containing the processed labels. 

    """
    
    #loop through the images in the folder
    sorted_images = sorted([f for f in os.listdir(images) if f.endswith('.nii') or f.endswith('.nii.gz')])
    
    #set the save folders
    if mode == "P":
        output_path = os.path.join(images, "Contrast_modified")
        
    elif mode == "T":
        image_output_path = os.path.join(images, "Augmented_images")
        label_output_path = os.path.join(labels, "Augmented_labels")
        
    for i in range(len(sorted_images)):
        image_name = sorted_images[i]
        image_path = os.path.join(images, image_name)
        
        #load the image data
        image = nib.load(image_path)
        ct_data = image.get_fdata()
        
        #call for contrast adjustment
        contrast_mod = contrastAdjustment(ct_data) #return the modified volume
        
        if mode == "P": 
            #save the contrast modified image
            output_file = os.path.join(output_path, f"mod_{image_name}")
            enhanced_nii = nib.Nifti1Image(contrast_mod, affine=np.eye(4))
            nib.save(enhanced_nii, output_file) 
            
        elif mode == "T":
            #get the labels
            sorted_labels = sorted([f for f in os.listdir(labels) if f.endswith('.nii') or f.endswith('.nii.gz')])
            label_name = sorted_labels[i]
            label_path = os.path.join(labels, label_name)
            
            label=nib.load(label_path)
            label_data = label.get_fdata()
            
            #call for augmentation
            aug_image, aug_label, orig_image, orig_label = volumeAugmentation(contrast_mod, label_data)  #return the volumes
            
            #save the augmented volumes (and the non-augmented, but ocntrast enchanced)
            #the goal is to douple the number of images -> save both
            image_output_file = os.path.join(image_output_path, f"aug_{image_name}")
            label_output_file = os.path.join(label_output_path, f"aug_{label_name}")
            orig_image_file = os.path.join(image_output_path, f"orig_{image_name}")
            orig_label_file = os.path.join(label_output_path, f"orig_{label_name}")
            
            image_nii = nib.Nifti1Image(aug_image, affine=np.eye(4))
            label_nii = nib.Nifti1Image(aug_label, affine=np.eye(4))
            orig_im_nii = nib.Nifti1Image(orig_image, affine=np.eye(4))
            orig_lab_nii = nib.Nifti1Image(orig_label, affine=np.eye(4))
            
            nib.save(image_nii, image_output_file) 
            nib.save(label_nii, label_output_file) 
            nib.save(orig_im_nii, orig_image_file)
            nib.save(orig_lab_nii, orig_label_file)
            
            output_path = [image_output_path, label_output_path]
            
    #return the path to processed data  
    return output_path
    
    
    

    

if __name__ == "__main__":
    
    BATCH_SIZE = 1
    
    acceptable_modes = ["T", "P", "C"]
    #does the user want to predict or tain the model
    mode = input("Do you want to train(T), predict(P) or close the program (C): ")
    
    while mode not in acceptable_modes:
        mode = input("The mode have to be T for train, P for predcit or C for closing: ")

    
    #find the gpu if available
    #define the device, gpu/cpu
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
     
        
    while mode == "T" or mode == "P":
        
        root_path, num_classes, images, labels, model_path, resume = set_parameters(mode)
    
        if mode ==  "T":
            #set the paths to folders
            train_images = os.path.join(root_path, images[0])
            train_labels = os.path.join(root_path, labels[0])
            
            val_images = os.path.join(root_path, images[1])
            val_labels = os.path.join(root_path, labels[1])
            
            #preprocess the images
            train_images, train_labels = pre_process(train_images, train_labels, mode)
            val_images, val_labels = pre_process(val_images, val_labels, mode)
            
            #prepare the datasets
            train_set = Dataset(images = train_images,
                                labels = train_labels,
                                num_classes = num_classes,
                                mode = mode)
          
            val_set = Dataset(images = val_images,
                              labels = val_labels,
                              num_classes = num_classes,
                              mode = mode)
            
            #create the dataloaders
            train_dataloader = DataLoader(dataset = train_set,
                                          batch_size = BATCH_SIZE,
                                          shuffle = True)
            
            val_dataloader = DataLoader(dataset = val_set,
                                        batch_size  = BATCH_SIZE,
                                        shuffle = True)
            
            
            training = ModuleTraining(train_dataloader, val_dataloader, 
                                      num_classes, BATCH_SIZE, model_path, 
                                      device, resume)
            training.print_results()
            
        else:
            #set the path to folders
            test_images = os.path.join(root_path, images)
            #just to clarify
            prediction_path = os.path.join(root_path, labels)
            
            #preprocess the images
            processed_images = pre_process(test_images, labels, mode)
            
            #prepare dataset
            test_set = Dataset(images = processed_images,
                               labels = "None",
                               num_classes = num_classes,
                               mode = mode)
            
            #prepare dataloader
            test_dataloader = DataLoader(dataset = test_set,
                                         batch_size = BATCH_SIZE,
                                         shuffle = True)
            
            #load the model
            model = UNet(1, num_classes)
            model.load_state_dict(torch.load(model_path, map_location = device))
            
            Predicting(test_dataloader, prediction_path, model, device)
            
        mode = input("Do you want to train(T), predict(P) or close the program (C): ")
            
        
        

