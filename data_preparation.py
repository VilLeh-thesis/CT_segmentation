#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 09:29:51 2025

@author: vilmalehto

The code for data preparation. The images and data must be in their own folders
under the same root folder. The images must be in NIfTI format. 

"""


import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset

class Dataset(Dataset):
    def __init__(self, images, labels, num_classes, mode):
        """
        The code creates a list structure to acces the images slice-by-slice.
        The list includes the name of the image file, the name of the label
        file, and the index of the wanted slice. 

        Parameters
        ----------
        root_path : String
            The path to the root folder.
        images : String
            The name of the image folder.
        labels : String
            The name of the image folder.
        num_classes : int
            The number of segmented classes. 
        mode : string, optional
            The default is "train". -> so far there is no changes to data
            preparation for train and predict data. 

        Returns
        -------
        None.

        """
        
        #could check that image and label path have corresponding files,
        #but let's skip that for now
        
        self.num_classes = num_classes
        self.mode = mode
        
        #get the full path to images
        self.image_path = images
        
        #for training we create the structure including images and labels 
        if self.mode == "T":
                
            #get the full path to labels
            self.label_path = labels
            
            #create sorted list of files in the end of path 
            #so a list of the image files and list of the label files, 
            #in corresponding order, due to the sorting
            self.image_files = sorted([f for f in os.listdir(self.image_path) if f.endswith('.nii') or f.endswith('.nii.gz')])
            self.label_files = sorted([f for f in os.listdir(self.label_path) if f.endswith('.nii') or f.endswith('.nii.gz')])
            
            #createas tuple of the files 
            #(image file name, corresponding label file name)
            pairs = zip(self.image_files, self.label_files)
                    
            #create a list of the slices and their corresponding masks
            #(the name of the image file, the name of the label file, and the index for the specific slice)
            self.slices = []
            
            for img_file, label_file in pairs:
                img_vol = nib.load(os.path.join(self.image_path, img_file)).get_fdata()
                
                #could check that the shapes are same, but I am not adding it yet
                #I am for now trusting that the image and label are same size
                
                #loop through the slices
                #but now also handle the label here
                for i in range(img_vol.shape[-1]):
                    self.slices.append((img_file, label_file, i))
                    
                    
        #for predicting, we don't have the labels
        #so it is handled differently (without the label files ofc)
        else:
            
            #create a sorted list of the image files
            self.image_files = sorted([f for f in os.listdir(self.image_path) if f.endswith('.nii') or f.endswith('.nii.gz')])
            
            #and a list of the image slices
            #the name of the image file and the index of the 2D slice
            self.image_slices = []
            
            #loop through all of the image files
            for img_file in self.image_files:
                #get the whole image
                img_vol = nib.load(os.path.join(self.image_path, img_file)).get_fdata()
                
                #save the image volume and index of the slice
                for i in range(img_vol.shape[-1]):
                    self.image_slices.append((img_file, i))
            
                
            
    def __len__(self):
        """
        Returns the length of the list.

        """
        if self.mode == "T":
            return len(self.slices)
        else:
            return len(self.image_slices)
    
    
    
    def __getitem__(self, index):
        """
        For training: function gets the specific image slice and the 
        correspondig mask. The function returns these. 
        
        For predicting: function gets the specific image slice. In addition,
        the whole image and the index of the slice are also returned as metadata. 

        Parameters
        ----------
        index : int
            The index of the wanted image.

        Returns
        -------
        image : torch.Tensor
            The 2D grayscale image.
        label : torch.tensor
            One-hot encoded segmentation mask of the corresponding image. 

        """
        if self.mode == "T":
            #get the infromation
            img_file, label_file, slice_idx = self.slices[index]
        
            #load the 3D volumes
            img_vol = nib.load(os.path.join(self.image_path, img_file)).get_fdata()
            label_vol = nib.load(os.path.join(self.label_path, label_file)).get_fdata()
            
            #get the slices (all pixels/voxels from the specific slice)
            #use np.float32 type to describe the intensity of the voxel/pixel 
            image = img_vol[:, :, slice_idx].astype(np.float32)
            label = label_vol[:, :, slice_idx].astype(np.float32)
            
            #normalize the grayscale values
            image =(image-np.min(image)) / (np.ptp(image) + 1e-8)
            
            # Add channel dimension for image: (1, H, W)
            image = torch.tensor(image).unsqueeze(0)
    
            # One-hot encode label: (C, H, W) (channels, height, width)
            label = torch.tensor(label, dtype=torch.long)
            label = F.one_hot(label, num_classes=self.num_classes)  # shape: (H, W, C)
            label = label.permute(2, 0, 1).float()  # shape: (C, H, W)
    
            return image, label
        
        else:
            image_file, slice_idx = self.image_slices[index]
            
            #get the whole image
            img_vol = nib.load(os.path.join(self.image_path, image_file)).get_fdata()
            #get the 2D image
            image = img_vol[:, :, slice_idx].astype(np.float32)
            
            #normalize the grayscale values
            image =(image-np.min(image)) / (np.ptp(image) + 1e-8)
            
            # Add channel dimension for image: (1, H, W)
            image = torch.tensor(image).unsqueeze(0)
            
            return image, image_file, slice_idx
        
            
        