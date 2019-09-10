#    UNet_main_train.py : Main Code for training the UNet accompanying publication "Classification of prostate cancer on MRI: Deep learning vs. clinical PI-RADS assessment", Patrick Schelb, Simon Kohl, Jan Philipp Radtke MD, Manuel Wiesenfarth PhD, Philipp Kickingereder MD, Sebastian Bickelhaupt, Tristan Anselm Kuder PhD, Albrecht Stenzinger, Markus Hohenfellner MD, Heinz-Peter Schlemmer MD, PhD, Klaus H. Maier-Hein PhD, David Bonekamp MD, Radiology, [manuscript accepted for publication]
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


import numpy as np
import pandas as pd
import os
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
from UNet_utils import train, save_checkpoint, adjust_learning_rate\
    ,BatchGenerator, CreateTrainValTestSplit, CrossEntropyLoss2d, get_class_frequencies, clear_image_data, \
      split_training, validate, \
    get_oversampling


####################################################################################################################
### Settings
####################################################################################################################

parser = argparse.ArgumentParser(description='RADIOLOGY-2019 Prostate Lesion Segmentation training script')

parser.add_argument('--input_file', default='', type=str, help='name of input file')
parser.add_argument('--output_path', default='', type=str, help='name of output folder')
parser.add_argument('--arch',default='UNet')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=80, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--b', default=32, type=int, help='mini-batch size')
parser.add_argument('--lr', default=0.00015, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
parser.add_argument('--num_splits', default=10, type=int, help='split dataset in 10 folds')
parser.add_argument('--num_val_folds', default=2, type=int, help='leave x fold out for validation')
parser.add_argument('--num_test_folds', default=2, type=int, help='leave x fold out for testing')
parser.add_argument('--seed', default=42, type=int, help='seed')
parser.add_argument('--patch_size', default=(160, 160), type=tuple, help='define patch size')
parser.add_argument('--p', default=1, type=float, help='probability for spatial transform')
parser.add_argument('--cv_number', default=16, type=int, help='number of cross validation loops')
parser.add_argument('--cv_start', default=0, type=int, help='')
parser.add_argument('--cv_end', default=16, type=int, help='')


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

    IDs = train_idx + val_idx

    print('size of training set {}'.format(len(train_idx)))
    print('size of validation set {}'.format(len(val_idx)))
    print('size of test set {}'.format(len(test_idx)))


    # data loading
    Data = ProstataData(args.input_file) #For details on this class see README

    # train and validate
    for cv in range(args.cv_start, args.cv_end):
        best_epoch = 0
        train_loss = []
        val_loss = []

        # define patients for training and validation
        train_idx, val_idx = split_training(IDs, len_val=62, cv=cv, cv_runs=args.cv_number)

        oversampling_factor, Slices_total, Natural_probability_tu_slice, Natural_probability_PRO_slice = get_oversampling(Data, train_idx=sorted(train_idx), Batch_Size=args.b, patch_size=args.patch_size)

        training_batches = Slices_total / args.b

        lr = args.lr
        base_lr = args.lr
        args.seed += 1

        print('train_idx', train_idx, len(train_idx))
        print('val_idx', val_idx, len(val_idx))

        # get class frequencies
        print('calculating class frequencie')

        Tumor_frequencie_ADC, Prostate_frequencie_ADC, Background_frequencie_ADC,\
        Tumor_frequencie_T2, Prostate_frequencie_T2, Background_frequencie_T2\
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

        # define model
        Net = UNetPytorch(in_shape=(3, args.patch_size[0], args.patch_size[1]))
        Net_Name = 'UNetPytorch'
        model = Net.cuda()

        # model parameter
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=args.weight_decay)
        criterion_ADC = CrossEntropyLoss2d(weight=torch.FloatTensor(weight_ADC)).cuda()
        criterion_T2 = CrossEntropyLoss2d(weight=torch.FloatTensor(weight_T2)).cuda()


        # new folder name for cv
        folder_name = general_folder_name + '/CV_{}'.format(cv)
        try:
            os.mkdir(folder_name)
        except OSError:
            pass

        checkpoint_file = folder_name + '/checkpoint_' + '{}.pth.tar'.format(Net_Name)

        # augmentation
        for epoch in range(args.epochs):
            torch.manual_seed(args.seed + epoch + cv)
            np.random.seed(epoch + cv)
            np.random.shuffle(train_idx)

            if epoch == 0:
                my_transforms = []
                spatial_transform = SpatialTransform(args.patch_size, np.array(args.patch_size) // 2,
                                                     do_elastic_deform=True, alpha=(100., 450.),
                                                     sigma=(13., 17.),
                                                     do_rotation=True, angle_z=(-np.pi / 2., np.pi / 2.),
                                                     do_scale=True, scale=(0.75, 1.25),
                                                     border_mode_data='constant', border_cval_data=0,
                                                     order_data=3,
                                                     random_crop=True)
                resample_transform = ResampleTransform(zoom_range=(0.7, 1.3))
                brightness_transform = BrightnessTransform(0.0, 0.1, True)
                my_transforms.append(resample_transform)
                my_transforms.append(ContrastAugmentationTransform((0.75, 1.25), True))
                my_transforms.append(brightness_transform)
                my_transforms.append(Mirror((2, 3)))
                all_transforms = Compose(my_transforms)
                sometimes_spatial_transforms = RndTransform(spatial_transform, prob=args.p,
                                                            alternative_transform=CenterCropTransform(
                                                                args.patch_size))
                sometimes_other_transforms = RndTransform(all_transforms, prob=1.0)
                final_transform = Compose([sometimes_spatial_transforms, sometimes_other_transforms])
                Center_Crop = CenterCropTransform(args.patch_size)

            if epoch == 30:
                my_transforms = []
                spatial_transform = SpatialTransform(args.patch_size, np.array(args.patch_size) // 2,
                                                     do_elastic_deform=True, alpha=(0., 250.),
                                                     sigma=(11., 14.),
                                                     do_rotation=True, angle_z=(-np.pi / 2., np.pi / 2.),
                                                     do_scale=True, scale=(0.85, 1.15),
                                                     border_mode_data='constant', border_cval_data=0,
                                                     order_data=3,
                                                     random_crop=True)
                resample_transform = ResampleTransform(zoom_range=(0.8, 1.2))
                brightness_transform = BrightnessTransform(0.0, 0.1, True)
                my_transforms.append(resample_transform)
                my_transforms.append(ContrastAugmentationTransform((0.85, 1.15), True))
                my_transforms.append(brightness_transform)
                all_transforms = Compose(my_transforms)
                sometimes_spatial_transforms = RndTransform(spatial_transform, prob=args.p,
                                                            alternative_transform=CenterCropTransform(
                                                                args.patch_size))
                sometimes_other_transforms = RndTransform(all_transforms, prob=1.0)
                final_transform = Compose([sometimes_spatial_transforms, sometimes_other_transforms])
                Center_Crop = CenterCropTransform(args.patch_size)

            if epoch == 50:
                my_transforms = []
                spatial_transform = SpatialTransform(args.patch_size, np.array(args.patch_size) // 2,
                                                     do_elastic_deform=True, alpha=(0., 150.),
                                                     sigma=(10., 12.),
                                                     do_rotation=True, angle_z=(-np.pi / 2., np.pi / 2.),
                                                     do_scale=True, scale=(0.85, 1.15),
                                                     border_mode_data='constant', border_cval_data=0,
                                                     order_data=3,
                                                     random_crop=False)
                resample_transform = ResampleTransform(zoom_range=(0.9, 1.1))
                brightness_transform = BrightnessTransform(0.0, 0.1, True)
                my_transforms.append(resample_transform)
                my_transforms.append(ContrastAugmentationTransform((0.95, 1.05), True))
                my_transforms.append(brightness_transform)
                all_transforms = Compose(my_transforms)
                sometimes_spatial_transforms = RndTransform(spatial_transform, prob=args.p,
                                                            alternative_transform=CenterCropTransform(
                                                                args.patch_size))
                sometimes_other_transforms = RndTransform(all_transforms, prob=1.0)
                final_transform = Compose([sometimes_spatial_transforms, sometimes_other_transforms])
                Center_Crop = CenterCropTransform(args.patch_size)


            train_loader = BatchGenerator(Data, BATCH_SIZE=args.b, split_idx=train_idx, seed=args.seed,
                                          ProbabilityTumorSlices=oversampling_factor, epoch=epoch,
                                          ADC_mean=ADC_mean, ADC_std=ADC_std, BVAL_mean=BVAL_mean, BVAL_std=
                                          BVAL_std, T2_mean=T2_mean, T2_std=T2_std)

            val_loader = BatchGenerator(Data, BATCH_SIZE=0, split_idx=val_idx, seed=args.seed,
                                     ProbabilityTumorSlices=None, epoch=epoch, test=True,
                                        ADC_mean=ADC_mean, ADC_std=ADC_std, BVAL_mean=BVAL_mean, BVAL_std=
                                        BVAL_std, T2_mean=T2_mean, T2_std=T2_std
                                        )


            #train on training set
            train_losses = train(train_loader=train_loader, model=model, optimizer=optimizer,
                                 criterion_ADC=criterion_ADC, criterion_T2=criterion_T2,
                                 final_transform=final_transform, workers=args.workers, seed=args.seed,
                                 training_batches=training_batches)
            train_loss.append(train_losses)


            # evaluate on validation set
            val_losses = validate(val_loader=val_loader, model=model, folder_name=folder_name,
                                  criterion_ADC =criterion_ADC, criterion_T2=criterion_T2, split_ixs=val_idx,
                                  epoch=epoch, workers=1, Center_Crop=Center_Crop, seed=args.seed)
            val_loss.append(val_losses)


            # write TrainingsCSV to folder name
            TrainingsCSV = pd.DataFrame({'train_loss': train_loss, 'val_loss': val_loss})
            TrainingsCSV.to_csv(folder_name + '/TrainingsCSV.csv')

            if val_losses <= min(val_loss):
                best_epoch = epoch
                print 'best epoch', epoch
                save_checkpoint({'epoch': epoch, 'arch': args.arch, 'state_dict': model.state_dict(),
                                 'optimizer': optimizer.state_dict()
                                 }, filename=checkpoint_file)

            optimizer, lr = adjust_learning_rate(optimizer, base_lr, epoch)

        # delete all output except best epoch
        clear_image_data(folder_name, best_epoch, epoch)


if __name__ == '__main__':
    main()


