# PROUNET
Prostate U-net

This repository provides the U-net training and testing script for the article "Classification of prostate cancer on MRI: Deep learning vs. clinical PI-RADS assessment", Patrick Schelb, Simon Kohl, Jan Philipp Radtke MD, Manuel Wiesenfarth PhD, Philipp Kickingereder MD, Sebastian Bickelhaupt, Tristan Anselm Kuder PhD, Albrecht Stenzinger, Markus Hohenfellner MD, Heinz-Peter Schlemmer MD, PhD, Klaus H. Maier-Hein PhD, David Bonekamp MD, Radiology, [manuscript accepted for publication]

# Link given in published article

The published article links to this repository through the website of the Division of Radiology (E010) of the German Cancer Research Center (DKFZ), at the following address: https://www.dkfz.de/de/radiologie/forschung/PROUNET.html.  Formally, the link given at the just named address determines the correct repository that should be linked to the article, please check there for updates.

# Input File
The input file should be a CSV file with all patients including their histopathology score, and the paths to the images and labels. All Information of Patient X can be accessed via the library Pandas with InputFile.iloc[X]

## Input Data (ProstataData Class)
Input Data for training, validation and testing should be a dictionary containing the following keys:

* `image`: 4D numpy array in shape (channels, x, y, z), with the channels containing the registered images of the modalities ADC, BVAL, T2 in the given order.

* `label`: 4D numpy array in shape (channels, x, y, z), with the channels containing the segmentations of the modalities Diffusion, T2 in the given order. The normal appearing prostate class is encoded as 1 in the segmentation mask whereas the tumor class (only significant cancer) is encoded as 2 in the segmentation mask.
 
Data[X] should return the described dictionary with the images and corresponding labels of Patient X by accessing the informations from the Input File. 

### UNet_main_train.py
To train a model ensemble use the script `UNet_main_train.py`.

Script outputs:
* TrainingsCSV.csv` file contains the training and validation loss

* `checkpoint_UNetPytorch.pth.tar` file contains the weights of the best model on the respective validation set (which can be used to predict the test set)

* `Val_Images` folder contains model outputs, masks and images for each patient in the validation set

#### UNet_main_test.py
To use the pertained model ensemble(UNet_main_train.py) for prediction of the test set use `UNet_main_test.py`.

Script outputs:
* TestCSV.csv` file contains the test loss

* `Test_Images` folder contains model outputs, masks and images for each patient in the validation set

Model outputs are then used to evaluate model performance.

##### UNet_net.py
Contains the python code for the UNet used in `UNet_main_train.py` and `UNet_main_test.py`.

##### UNet_utils.py
Contains the python code for the functions used in `UNet_main_train.py` and `UNet_main_test.py`.






