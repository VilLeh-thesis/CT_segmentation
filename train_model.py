#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 12:59:12 2025

@author: vilmalehto

The code for model training and predicting. These functions are called by 
model interface when neededn. 


For training: The code trains the model and saves it at the end of the path 
defined by user.

To train the model you need to know the path for training data and validation 
data. These have to be under the same root folder. 


For predicting: 

"""

import torch
import os
import nibabel as nib
import numpy as np
from torch import optim, nn
from tqdm import tqdm
import torch.nn.functional as F


from unet import UNet #our unet class

class HybridLoss(nn.Module):
    def __init__(self, weight_ce=0.7, weight_dice=0.3, eps=1e-6):
        """
        The Loss-function used in model training. This is a mix of Dice loss 
        and Croos-Entropy loss functions. A loss function computes the loss 
        between model predictions and ground-truth, therefore used to guide
        model training.

        Parameters
        ----------
        weight_ce : optional, float
            The default is 0.7. Describes how high the Cross 
            entropy loss is valued in hybrid loss.
        weight_dice : optional, float
            The default is 0.3. Describes how high the Dice loss 
            is valued in hybrid loss. 
        eps : optional, float
            The default is 1e-6. Small constant added to the 
            denominator for numerical stability and to avoid division by zero.

        Returns
        -------
        None.

        """
        super(HybridLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.eps = eps
        self.ce_loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, result, target):
        """
        Forward function to count the loss function outcome. 

        Parameters
        ----------
        result : torch.Tensor
            The result of the model prediction, in shape of (B, C, H, W). 
            Batch size, number of Classes, Height, Width
        target : torch.Tensor
            The correct result from ground-truth mask, in shape of (B, C, H, W).

        Returns
        -------
        hybrid_loss : torch.Tensor
            Value combining Dice loss and Cross-Entropy loss.

        """
        
        # Dice loss
        probs = F.softmax(result, dim=1)
        intersection = (probs * target).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
        dice_per_class = (2 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1 - dice_per_class.mean()
        
        # Cross-entropy loss
        ce_target = target.argmax(dim=1).long()
        ce_loss = self.ce_loss_fn(result, ce_target)
        
        # Combine and return the result value
        hybrid_loss = self.weight_dice * dice_loss + self.weight_ce * ce_loss
        return hybrid_loss
    
    
    
class ModuleTraining():
    def __init__(self, train_dataloader, val_dataloader, num_classes, batch_size, model_path, device, resume):
        """
        Functions trains and validates the model. At the end the model is saved
        to a location specified by the user. 

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            DataLoader providing batches of input images and corresponding
            ground-truth segmentation masks from training dataset. 
        val_dataloader : torch.utils.data.DataLoader
            DataLoader providing batches of input images and corresponding
            ground-truth segmentation masks from validation dataset.
        num_classes : int
            The number of segmented classes, also count background.
        batch_size : int
            The number of images in one batch.
        model_path : string
            The path to where the model is saved after training.
        resume: string
            Tells if the training is continued from previous session. 
            Y = a previous model training is continued.

        Returns
        -------
        None.

        """
    
        #set the learning parameters
        LEARNING_RATE = 0.001
        EPOCHS = 1
            
        #set the parameters given by user
        self.train_images = train_dataloader
        self.val_images = val_dataloader
        self.num_classes = num_classes
        
        #if you  want to save multiple models change the name of the file below
        #inbetween the runs
        #model_path is the path to folder where the trained models are saved
        self.model_path = os.path.join(model_path, "unet.pth")
        #checkpoint_path is path to folder where only partially trained models are
        #this is not mandatory, you can also handle everything through one file
        #or train the model just once
        self.checkpoint_path = os.path.join(model_path, "checkpoints", 
                                            "model_v1.pth")
        
        #define the model
        #in_channels = 1 for grayscale and 3 for RGB
        #num_classes = 5: background, skull, CSF, brain and haemorrhage
        model = UNet(in_channels = 1, num_classes = self.num_classes).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
        
        criterion = HybridLoss() #sets up the loss function, also this 
        
        #incase of resuming to old model training, reload the model and optimizer
        if resume == "Y":
          checkpoint = input("What is the path to previous training session data? (.pth file) ")
          checkpoint = torch.load(checkpoint, map_location = device)
          
          model.load_state_dict(checkpoint["model_state"])
          optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        #init a list for prediction results
        self.results = []
        
        #the training
        for epoch in tqdm(range(EPOCHS)):
            model.train()
            train_running_loss = 0
            
            #loop through training data
            for idx, img_mask in enumerate(tqdm(train_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                #predict the segments
                y_pred = model(img)
                optimizer.zero_grad()
                
                #count the loss for the prediction
                loss = criterion(y_pred, mask) #compare the manual segmentation and prediction
                train_running_loss += loss.item()
                
                #optimize the weights based on the loss function results 
                loss.backward()
                optimizer.step()
                
            #onto the validation
            train_loss = train_running_loss / idx+1
            
            model.eval()
            val_runnin_loss = 0
            with torch.no_grad():
                #loop through validation data
                for idx, img_mask in enumerate(tqdm(val_dataloader)):
                    img = img_mask[0].float().to(device)
                    mask = img_mask[1].float().to(device)
                    
                    #prediction
                    y_pred = model(img)
                    
                    #count the valdiation loss
                    loss = criterion(y_pred, mask)
                    val_runnin_loss += loss.item()
                    
                val_loss = val_runnin_loss / idx+1
                
            #print the stats for every epoch
            print()
            print("-"*30) 
            print(f"EPOCH {epoch +1 }")
            print(f"Train loss {train_loss:.4f}")
            print(f"Validation loss {val_loss:.4f}")
            
            self.results.append((("training", train_loss), ("valdiation", val_loss)))
        
        #at the end just save the model, no need to return that
        #save the model and the optimizer, incase the training will be continued
        torch.save({"model_state": model.state_dict(), 
                    "optimizer_state": optimizer.state_dict()}, self.checkpoint_path)
        
        #also save model
        #this is the one used when the training is not continued
        #and the model is only used for prediction
        torch.save(model.state_dict(), self.model_path)

    def print_results(self):
        """
        Function for printing the results. 

        Returns
        -------
        None.

        """
        print(self.results)
        
    
    
    
      
class Predicting():
    def __init__(self, images, prediction_path, model, device):
        """
        A class for predicting with the model. 
        
        The model is set to eval state and the gradient compunation is turned off.
        The program loops through the slices and uses the argmax() to get the class
        with highest likelihood. The newly created prediction and the metadata
        form the original image are saved to dictionaries, form where those
        are saved in .nii format to folder defined by user. 

        Parameters
        ----------
        images: DataLoader
            DataLoader including the images to be segmented. 
        prediction_path: String
            The path to the file where the segmentation masks will be saved. 
        model : PyTorch - CNN network
            The custom UNet model done based on the original UNet archicture 
            (Ronneberger et al., 2015). In this code it is set to handle grayscale 
            images.
        device: 
            The device the model is run on (CPU/GPU)
    
        Returns
        -------
        None.
    
        """
        
        #set the model to evaluation state
        model.eval()
        model.to(device)
        
        #createa dictionary for predictions
        volumes = {}
        metadata = {}
        
        #predicting
        with torch.no_grad():
            
            #loop through the slices
            #the image slice, the name of the image file, and the index of the image slice
            for image, img_file, slice_idx in tqdm(images, desc="predict"):
                #get the 2D image to gpu if available
                image = image.to(device)
                #pass the image through the model
                output = model(image)
                #get the prediction
                #argmax() gets the most likely class per pixel/voxel
                pred = torch.argmax(output, dim=1).cpu().numpy().squeeze(0) #2D numpy array
                
                #extract the filename and slice index
                img_file = img_file[0]
                slice_idx = slice_idx.item()
                
                #check if the file already exist in volumes dict
                if img_file not in volumes:
                    #if not create it
                    volumes[img_file] = {}
                    
                    #get the metadata from original file
                    path = os.path.join(images.dataset.image_path, img_file)
                    nii = nib.load(path)
                    metadata[img_file] = (nii.affine, nii.header)
                    
                #save the prediction to correct spot
                volumes[img_file][slice_idx] = pred
                    #volumes = { "img_file_1.nii": {0: pred, 1: pred}, 
                    #           "img_file_2.nii" = {0: pred, 1: pred}}
                
        #loop through the predictions
        for img_file, slice_preds in volumes.items():
            #Find the highest slice number for the specific image file
            max_idx = max(slice_preds.keys()) + 1
            #initialize the array for 3D label
            volume = np.zeros((list(slice_preds.values())[0].shape[0],
                               list(slice_preds.values())[0].shape[1],
                               max_idx), dtype=np.uint8)
            
            #get each prediction slice
            for idx, prediction in slice_preds.items():
                #save the slice to its correct spot in the 3D array
                volume[:, :, idx] = prediction
            
            #get the corresponding metadata
            affine, header = metadata[img_file]
            #create the .nii format with prediction volume and metadata
            pred_nii = nib.Nifti1Image(volume, affine, header)
            
            #save the prediction to file corresponding to image file
            prediction_file = "head_mask_"+img_file
            save_path = os.path.join(prediction_path, prediction_file)
            nib.save(pred_nii, save_path)    
        
            
    