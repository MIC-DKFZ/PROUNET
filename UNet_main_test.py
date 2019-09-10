#    UNet_main_test.py : Main Code for testing the UNet accompanying publication "Classification of prostate cancer on MRI: Deep learning vs. clinical PI-RADS assessment", Patrick Schelb, Simon Kohl, Jan Philipp Radtke MD, Manuel Wiesenfarth PhD, Philipp Kickingereder MD, Sebastian Bickelhaupt, Tristan Anselm Kuder PhD, Albrecht Stenzinger, Markus Hohenfellner MD, Heinz-Peter Schlemmer MD, PhD, Klaus H. Maier-Hein PhD, David Bonekamp MD, Radiology, [manuscript accepted for publication]
#    Copyright (C) 2019  German Cancer Research Center (DKFZ)

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

#    contact: David Bonekamp, MD, d.bonekamp@dkfz-heidelberg.de

__author__  = "German Cancer Research Center (DKFZ)"


import os
import numpy as np
import pandas as pd
import torch
import argparse
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, Mirror
from batchgenerators.transforms.noise_transforms import RicianNoiseTransform
from batchgenerators.transforms.resample_transforms import ResampleTransform
from batchgenerators.transforms.abstract_transforms import RndTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform
from batchgenerators.transforms.sample_normalization_transforms import CutOffOutliersTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform,  BrightnessTransform, ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.sample_normalization_transforms import MeanStdNormalizationTransform
from UNet_net import UNetPytorch
from UNet_utils import BatchGenerator, CreateTrainValTestSplit, validate, get_class_frequencies, CrossEntropyLoss2d


####################################################################################################################
### Settings
####################################################################################################################

parser = argparse.ArgumentParser(description='RADIOLOGY-2019 Prostate Lesion Segmentation ensemble testing script')

parser.add_argument('--input_file', default='', type=str, help='name of input file')
parser.add_argument('--ensemble_path', default='', type=str, help='path to pretrained CNN ensemble')
parser.add_argument('--checkpoint_name', default='checkpoint_UNetPytorch.pth.tar', type=str, help='name of ensemble checkpoint files')
parser.add_argument('--output_path', default='', type=str, help='name of output folder')
parser.add_argument('--num_splits', default=10, type=int, help='split dataset in 10 folds')
parser.add_argument('--num_val_folds', default=2, type=int, help='leave x fold out for validation')
parser.add_argument('--num_test_folds', default=2, type=int, help='leave x fold out for testing')
parser.add_argument('--seed', default=42, type=int, help='seed')
parser.add_argument('--patch_size', default=(160, 160), type=tuple, help='define patch size')


def main():

    # assign global args
    global args
    args = parser.parse_args()


    # make a folder for the experiment
    general_folder_name = args.output_path
    try:
        os.mkdir(general_folder_name)
    except OSError:
        pass

    # create train, test split, return the indices, patients in test_split wont be seen during whole training
    train_idx, val_idx, test_idx = CreateTrainValTestSplit(HistoFile_path=args.input_file, num_splits=args.num_splits, num_test_folds=args.num_test_folds,
                                        num_val_folds=args.num_val_folds, seed=args.seed)

    print('size of training set {}'.format(len(train_idx)))
    print('size of validation set {}'.format(len(val_idx)))
    print('size of test set {}'.format(len(test_idx)))

    train_idx = train_idx + val_idx

    # data loading
    Data = ProstataData(args.input_file) #For details on this class see README


    # get class frequencies
    print('calculating class frequencie')

    Tumor_frequencie_ADC, Prostate_frequencie_ADC, Background_frequencie_ADC, \
    Tumor_frequencie_T2, Prostate_frequencie_T2, Background_frequencie_T2 \
        , ADC_mean, ADC_std, BVAL_mean, BVAL_std, T2_mean, T2_std \
        = get_class_frequencies(Data, train_idx, patch_size=args.patch_size)

    print ADC_mean, ADC_std, BVAL_mean, BVAL_std, T2_mean, T2_std

    print('ADC', Tumor_frequencie_ADC, Prostate_frequencie_ADC, Background_frequencie_ADC)
    print('T2', Tumor_frequencie_T2, Prostate_frequencie_T2, Background_frequencie_T2)

    all_ADC = np.float(Background_frequencie_ADC + Prostate_frequencie_ADC + Tumor_frequencie_ADC)
    all_T2 = np.float(Background_frequencie_T2 + Prostate_frequencie_T2 + Tumor_frequencie_T2)

    print all_ADC
    print all_T2

    W1_ADC = 1 / (Background_frequencie_ADC / all_ADC) ** 0.25
    W2_ADC = 1 / (Prostate_frequencie_ADC / all_ADC) ** 0.25
    W3_ADC = 1 / (Tumor_frequencie_ADC / all_ADC) ** 0.25

    Wa_ADC = W1_ADC / (W1_ADC + W2_ADC + W3_ADC)
    Wb_ADC = W2_ADC / (W1_ADC + W2_ADC + W3_ADC)
    Wc_ADC = W3_ADC / (W1_ADC + W2_ADC + W3_ADC)

    print 'Weights ADC', Wa_ADC, Wb_ADC, Wc_ADC

    weight_ADC = (Wa_ADC, Wb_ADC, Wc_ADC)

    W1_T2 = 1 / (Background_frequencie_T2 / all_T2) ** 0.25
    W2_T2 = 1 / (Prostate_frequencie_T2 / all_T2) ** 0.25
    W3_T2 = 1 / (Tumor_frequencie_T2 / all_T2) ** 0.25

    Wa_T2 = W1_T2 / (W1_T2 + W2_T2 + W3_T2)
    Wb_T2 = W2_T2 / (W1_T2 + W2_T2 + W3_T2)
    Wc_T2 = W3_T2 / (W1_T2 + W2_T2 + W3_T2)

    print 'Weights T2', Wa_T2, Wb_T2, Wc_T2

    weight_T2 = (Wa_T2, Wb_T2, Wc_T2)

    criterion_ADC = CrossEntropyLoss2d(weight=torch.FloatTensor(weight_ADC)).cuda()
    criterion_T2 = CrossEntropyLoss2d(weight=torch.FloatTensor(weight_T2)).cuda()

    Center_Crop = CenterCropTransform(args.patch_size)
    count = 0

    for folder, subfolders, files in sorted(os.walk(args.ensemble_path)):
        for file in files:
            if file.startswith(args.checkpoint_name):
                epoch = 0

                # define model
                Net = UNetPytorch(in_shape=(3, args.patch_size[0], args.patch_size[1]))
                model = Net.cuda()

                folder_name_for_single_nets = general_folder_name + '/Net_{}'.format(count)

                try:
                    os.mkdir(folder_name_for_single_nets)
                except OSError:
                    pass

                model_path = os.path.join(os.path.abspath(folder), file)
                checkpoint = torch.load(model_path)
                model.load_state_dict(checkpoint['state_dict'])
                test_loader = BatchGenerator(Data, BATCH_SIZE=0, split_idx=test_idx, seed=args.seed,
                                             ProbabilityTumorSlices=None, epoch=epoch, test=True,
                                             ADC_mean=ADC_mean, ADC_std=ADC_std, BVAL_mean=BVAL_mean, BVAL_std=
                                             BVAL_std, T2_mean=T2_mean, T2_std=T2_std
                                             )

                test_loss = validate(test_loader, model,
                    folder_name=folder_name_for_single_nets, criterion_ADC=criterion_ADC, criterion_T2=criterion_T2,
                    split_ixs=test_idx, epoch=epoch, workers=1, Center_Crop=Center_Crop, seed=args.seed, test=True)

                TestCSV = pd.DataFrame({'test_loss': [test_loss]})
                TestCSV.to_csv(folder_name_for_single_nets + '/TestCSV.csv')
                count += 1

if __name__ == '__main__':
    main()




