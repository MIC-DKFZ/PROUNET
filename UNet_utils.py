#    UNet_utils.py : Utilities Code (function definitions) accompanying publication "Classification of prostate cancer on MRI: Deep learning vs. clinical PI-RADS assessment", Patrick Schelb, Simon Kohl, Jan Philipp Radtke MD, Manuel Wiesenfarth PhD, Philipp Kickingereder MD, Sebastian Bickelhaupt, Tristan Anselm Kuder PhD, Albrecht Stenzinger, Markus Hohenfellner MD, Heinz-Peter Schlemmer MD, PhD, Klaus H. Maier-Hein PhD, David Bonekamp MD, Radiology, [manuscript accepted for publication]
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

import pandas as pd
import os
import SimpleITK as sitk
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import DataLoaderBase
from builtins import object
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import shutil
import nrrd
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop
from batchgenerators.transforms.sample_normalization_transforms import CutOffOutliersTransform
from batchgenerators.augmentations.normalizations import cut_off_outliers


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, reduce=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight=weight, size_average=size_average, reduce=reduce)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


def ToTensor(batch):

    image, label = batch['data'], batch['seg']

    data = torch.from_numpy(image[:,:,:,:])
    seg = torch.from_numpy(label[:,0,:,:])
    seg_T2 = torch.from_numpy(label[:,1,:,:])

    return {'data': data,
            'seg': seg,
            'seg_T2': seg_T2}



def train(train_loader, model, optimizer, criterion_ADC, criterion_T2, final_transform, workers, seed,
          training_batches):

    train_losses = AverageMeter()
    np.random.seed(seed)
    seeds = np.random.choice(seed, workers, False, None)
    model.train()
    multithreaded_generator = MultiThreadedAugmenter(train_loader, final_transform, workers, 2, seeds=seeds)
    torch.cuda.empty_cache()

    for i in range(training_batches):
        print('Batch: [{0}/{1}]'.format(i +1, training_batches))
        batch = multithreaded_generator.next()
        TensorBatch = ToTensor(batch)
        target = TensorBatch['seg'].cuda()
        target_T2 = TensorBatch['seg_T2'].cuda()
        input_var = torch.autograd.Variable(TensorBatch['data'], requires_grad=True).cuda(async=True)
        input_var = input_var.float()
        target_var = torch.autograd.Variable(target)
        target_var = target_var.long()
        target_var_T2 = torch.autograd.Variable(target_T2)
        target_var_T2 = target_var_T2.long()
        optimizer.zero_grad()
        output = model(input_var)
        loss_ADC = criterion_ADC(output, target_var)
        loss_T2 = criterion_T2(output, target_var_T2)
        loss = (loss_ADC + loss_T2) / 2.
        loss.backward()
        optimizer.step()
        train_losses.update(loss.item())
        print 'train_loss', loss.item()

    torch.cuda.empty_cache()

    return train_losses.avg



def validate(val_loader, model, epoch, criterion_ADC, criterion_T2 ,split_ixs, Center_Crop, workers, seed, folder_name,
             test=False):

    val_losses = AverageMeter()
    seeds = np.random.choice(seed, workers, False, None)
    torch.cuda.empty_cache()
    model.eval()
    multithreaded_generator = MultiThreadedAugmenter(val_loader, Center_Crop, workers, 2, seeds=seeds)

    for i in range(len(split_ixs)):
        patient = split_ixs[i]
        print 'patient', patient
        batch = multithreaded_generator.next()
        TensorBatch = ToTensor(batch)
        target = TensorBatch['seg'].cuda()
        target_T2 = TensorBatch['seg_T2'].cuda()
        input_var = torch.autograd.Variable(TensorBatch['data'], volatile=True).cuda(async=True)
        input_var = input_var.float()
        target_var = torch.autograd.Variable(target, volatile=True)
        target_var = target_var.long()
        target_var_T2 = torch.autograd.Variable(target_T2, volatile=True)
        target_var_T2 = target_var_T2.long()
        output = model(input_var)
        probs = F.softmax(output)
        loss_ADC = criterion_ADC(output, target_var)
        loss_T2 = criterion_T2(output, target_var_T2)
        loss = (loss_ADC + loss_T2) / 2.
        val_losses.update(loss.item())
        if test == False:
            print 'val_loss', loss.item()
        else:
            print 'test_loss', loss.item()

        image = (input_var.data).cpu().numpy()
        Mprobs = (probs.data).cpu().numpy()
        fprobs = (probs.data).cpu().numpy()
        segmentation = (target).cpu().numpy()
        segmentation_T2 = (target_T2).cpu().numpy()
        label = np.where(segmentation == 2, 1, 0)
        label_T2 = np.where(segmentation_T2 == 2, 1, 0)
        label = np.uint8(label)
        label_T2 = np.uint8(label_T2)
        PRO = np.where(segmentation == 1, 1, 0)
        PRO = np.uint8(PRO)
        PRO_T2 = np.where(segmentation_T2 == 1, 1, 0)
        PRO_T2 = np.uint8(PRO_T2)

        fprobs[:, 0, :, :] = fprobs[:, 0, :, :] == np.amax(fprobs, axis=1)
        fprobs[:, 1, :, :] = fprobs[:, 1, :, :] == np.amax(fprobs, axis=1)
        fprobs[:, 2, :, :] = fprobs[:, 2, :, :] == np.amax(fprobs, axis=1)

        ProstateOut = fprobs[:, 1, :, :]
        TumorOut = fprobs[:, 2, :, :]

        probability_map_back = Mprobs[:, 0, :, :]
        probability_map_pro = Mprobs[:, 1, :, :]
        probability_map_tu = Mprobs[:, 2, :, :]

        if test == False:
            try:
                os.mkdir(folder_name + '/Val_Images')
            except OSError:
                pass

            try:
                os.mkdir(folder_name + '/Val_Images/Epoch_{}'.format(epoch))
            except OSError:
                pass

            save_images_to = folder_name + '/Val_Images/Epoch_{}/Patient_{}'.format(epoch, patient)
            try:
                os.mkdir(save_images_to)
            except OSError:
                pass

        else:
            try:
                os.mkdir(folder_name + '/Test_Images')
            except OSError:
                pass

            save_images_to = folder_name + '/Test_Images/Patient_{}'.format(patient)
            try:
                os.mkdir(save_images_to)
            except OSError:
                pass


        ADCimage = image[:, 0, :, :]
        BVALimage = image[:, 1, :, :]
        T2image = image[:, 2, :, :]

        TumorOut = sitk.GetImageFromArray(np.uint8(TumorOut))
        ProstateOut = sitk.GetImageFromArray(np.uint8(ProstateOut))
        ADCimg = sitk.GetImageFromArray(ADCimage)
        BVALimg = sitk.GetImageFromArray(BVALimage)
        T2img = sitk.GetImageFromArray(T2image)
        seg = sitk.GetImageFromArray(label)
        seg_T2 = sitk.GetImageFromArray(label_T2)
        pro = sitk.GetImageFromArray(PRO)
        pro_T2 = sitk.GetImageFromArray(PRO_T2)
        probsBack = sitk.GetImageFromArray(probability_map_back)
        probsPRO = sitk.GetImageFromArray(probability_map_pro)
        probsTU = sitk.GetImageFromArray(probability_map_tu)

        save(TumorOut, save_images_to + '/Tumor_Output.nrrd', Mask=True)
        save(ProstateOut, save_images_to + '/Prostate_Output.nrrd', Mask=True)
        save(ADCimg, save_images_to + '/ADCImage.nrrd')
        save(BVALimg, save_images_to + '/BVALImage.nrrd')
        save(T2img, save_images_to + '/T2Image.nrrd')

        save(seg, save_images_to + '/Label.nrrd', Mask=True)
        save(seg_T2, save_images_to + '/Label_T2.nrrd', Mask=True)
        save(pro, save_images_to + '/Pro_Label.nrrd', Mask=True)
        save(pro_T2, save_images_to + '/Pro_Label_T2.nrrd', Mask=True)
        save(probsBack, save_images_to + '/ProbabilityMapBack.nrrd')
        save(probsTU, save_images_to + '/ProbabilityMapTU.nrrd')
        save(probsPRO, save_images_to + '/ProbabilityMapPRO.nrrd')
        torch.cuda.empty_cache()

    return val_losses.avg


def clear_image_data(folder_name, best_epoch, epoch):
    for e in range(epoch+1):
        if e == best_epoch:
            print('best epoch')
        else:
            path_name = folder_name + '/Val_Images/Epoch_{}/'.format(e)
            try:
                shutil.rmtree(path_name)
            except OSError:
                pass


def save_checkpoint(state, filename):
    torch.save(state, filename)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, arg_lr, epoch):

    lr = np.float32(arg_lr * 0.98 ** epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer, lr


def resample(fixed, target_resample_resolution, Mask=False):

    if Mask == True:
        Interpolator = sitk.sitkNearestNeighbor
    else:
        Interpolator = sitk.sitkBSpline

    fixed_spacing = fixed.GetSpacing()
    fixed_spacing = np.array(fixed_spacing)
    fixed_size = fixed.GetSize()
    fixed_size = np.array(fixed_size)

    image_size = fixed_size * (fixed_spacing / target_resample_resolution)
    image_size = np.around(image_size)
    image_size = image_size.astype(np.uint32).tolist()

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_resample_resolution)
    resample.SetInterpolator(Interpolator)
    resample.SetOutputOrigin(fixed.GetOrigin())
    resample.SetOutputDirection(fixed.GetDirection())
    resample.SetSize(image_size)

    out = resample.Execute(fixed)

    return out


class BatchGenerator(DataLoaderBase):

    def __init__(self, data, BATCH_SIZE, split_idx, seed, ADC_mean, ADC_std, BVAL_mean, BVAL_std, T2_mean, T2_std,
                 ProbabilityTumorSlices=None, epoch=None, test=False):
        super(self.__class__, self).__init__(data=data, BATCH_SIZE=BATCH_SIZE, seed=False, num_batches=None)
        self._split_idx = split_idx
        self._ProbabilityTumorSlices = ProbabilityTumorSlices
        self._epoch = epoch
        self._s = 0
        self._count = 0
        self.test = test
        self.seed = seed
        self.ADC_mean = ADC_mean
        self.BVAL_mean = BVAL_mean
        self.T2_mean = T2_mean
        self.ADC_std = ADC_std
        self.BVAL_std = BVAL_std
        self.T2_std = T2_std

    def generate_train_batch(self):

        channels_img = 3
        channels_label = 2
        img_size = 200

        if self.test == True:

            img = np.empty((self.BATCH_SIZE, channels_img, img_size, img_size))
            label = np.empty((self.BATCH_SIZE, channels_label, img_size, img_size))

            if self._count < len(self._split_idx):
                idx = self._split_idx[self._count]
                z_dim = self._data[idx]['image'].shape[3]
                self.BATCH_SIZE = z_dim

                custom_batch = (z_dim / self.BATCH_SIZE)*self.BATCH_SIZE


                if self._count < len(self._split_idx):
                    img = self._data[idx]['image']
                    img = img.transpose((3,0,1,2))
                    label = self._data[idx]['label']
                    label = label.transpose((3,0,1,2))
                    self._count += 1


                else:
                    for b in range(self.BATCH_SIZE):
                        if self._s < custom_batch:
                            img[b, :, :, :] = self._data[idx]['image'][:channels_img, :, :, self._s]
                            label[b, :, :, :] = self._data[idx]['label'][:channels_label, :, :, self._s]
                            self._s += 1
                        else:
                            self._s = 0
                            self._count += 1
                            self._batches_generated = self._count
                            if self._count < len(self._split_idx):
                                idx = self._split_idx[self._batches_generated]
                                img[b, :, :, :] = self._data[idx]['image'][:channels_img, :, :, self._s]
                                label[b, :, :, :] = self._data[idx]['label'][:channels_label, :, :, self._s]
                                self._s += 1
                            else:
                                pass

            else:
                pass

        else:

            idx = np.random.choice(self._split_idx, self.BATCH_SIZE, False, None)
            img = np.empty((self.BATCH_SIZE, channels_img, img_size, img_size))
            label = np.empty((self.BATCH_SIZE, channels_label, img_size, img_size))

            for b in range(self.BATCH_SIZE):

                if self._ProbabilityTumorSlices is not None:
                    LabelData = self._data[idx[b]]['label']
                    z_dim = self._data[idx[b]]['image'].shape[3]
                    CancerSlices = []
                    for Slice in range(z_dim):

                        bool = np.where(LabelData[:, :, :, Slice] == 2, True, False)

                        if bool.any() == True:
                            CancerSlices.append(Slice)

                    if sum(CancerSlices) is not 0:
                        CancerSlices = np.array(CancerSlices)
                        totalTumorSliceProb = float(self._ProbabilityTumorSlices)
                        totalOtherSliceProb = float(1) - float(totalTumorSliceProb)

                        ProbabilityMap = np.array(np.zeros(z_dim))
                        ProbabilityMap[CancerSlices] = self._ProbabilityTumorSlices / float(len(CancerSlices))

                        NoTumorSlices = float(z_dim - len(CancerSlices))
                        ProbPerNoTumorSlice = totalOtherSliceProb / NoTumorSlices

                        ProbabilityMap = [ProbPerNoTumorSlice if g == 0 else g for g in ProbabilityMap]

                    else:
                        ProbabilityMap = np.zeros(z_dim)
                        ProbabilityMap = [float(1) / float(z_dim) if g == 0 else g for g in ProbabilityMap]

                else:
                    ProbabilityMap = np.zeros(z_dim)
                    ProbabilityMap = [float(1) / float(z_dim) if g == 0 else g for g in ProbabilityMap]

                randint = np.random.choice(z_dim, p=ProbabilityMap)

                img[b, :, :, :] = self._data[idx[b]]['image'][:, :, :, randint]
                label[b, :, :, :] = self._data[idx[b]]['label'][:, :, :, randint]

        # cut off outliers before image normalization

        img = np.nan_to_num(img)
        img = cut_off_outliers(img, percentile_lower=0.2, percentile_upper=99.8, per_channel=True)

        img[:, 0, :, :] = (img[:, 0, :, :] - np.float(self.ADC_mean)) / np.float(self.ADC_std)
        img[:, 1, :, :] = (img[:, 1, :, :] - np.float(self.BVAL_mean)) / np.float(self.BVAL_std)
        img[:, 2, :, :] = (img[:, 2, :, :] - np.float(self.T2_mean)) / np.float(self.T2_std)



        img = np.float32(img)

        data_dict = {"data": img, "seg": label}


        return data_dict




def CreateTrainValTestSplit(HistoFile_path, num_splits, num_val_folds, num_test_folds, seed):

    np.random.seed(seed)

    HistoFile = pd.read_csv(HistoFile_path)

    # calculate random split assignments of the subjects
    IDs = HistoFile.Master_ID.values # Patient no.
    unique_IDs = np.unique(IDs)

    num_subjects = len(unique_IDs)
    splits = -1 * np.ones(num_subjects)
    s_per_split = num_subjects // num_splits
    # assign an equivalent # subjects to the splits
    assign = np.random.choice(range(num_subjects), size=(num_splits, s_per_split), replace=False)
    for split in range(num_splits):
        for subj in range(s_per_split):
            splits[assign[split, subj]] = split

    # assign missing subjects
    ixs = np.where(splits == -1)
    splits[ixs] = np.random.randint(0, high=num_splits, size=len(ixs[0]))

    train_splits = [s for s in range(num_splits)]
    subjects = []
    for s in train_splits:
        split_ixs = np.where(splits == s)

        split_subjs = unique_IDs[split_ixs]
        subjects.append(split_subjs)

    train_folds = [f for f in range(num_splits - num_val_folds - num_test_folds)]
    train_ixs_lists = [subjects[fold] for fold in train_folds]
    train_ixs = [ix for ixs in train_ixs_lists for ix in ixs]  # flatten the list of lists

    val_folds = [f for f in range(num_splits - num_test_folds) if f not in train_folds]
    val_ixs_lists = [subjects[fold] for fold in val_folds]
    val_ixs = [ix for ixs in val_ixs_lists for ix in ixs]  # flatten the list of lists

    test_folds = [f for f in range(num_splits) if f not in val_folds + train_folds]
    test_ixs_lists = [subjects[fold] for fold in test_folds]
    test_ixs = [ix for ixs in test_ixs_lists for ix in ixs]  # flatten the list of lists


    return train_ixs, val_ixs, test_ixs


def get_class_frequencies(Data, train_idx, patch_size):

    Tumor_frequencie_ADC = 0
    Prostate_frequencie_ADC = 0
    Background_frequencie_ADC = 0

    Tumor_frequencie_T2 = 0
    Prostate_frequencie_T2 = 0
    Background_frequencie_T2 = 0

    T2_mean = 0
    T2_std = 0
    ADC_mean = 0
    ADC_std = 0
    BVAL_mean = 0
    BVAL_std = 0

    for i in range(len(train_idx)):
        idx = train_idx[i]
        Data_class = Data[idx]['label']
        Data_image = Data[idx]['image']

        center_crop_dimensions = ((patch_size[0], patch_size[1]))

        Data_class_Label = center_crop(Data_class, center_crop_dimensions)
        Data_class_Label = Data_class_Label[0]

        Data_image_cropped = center_crop(Data_image, center_crop_dimensions)
        Data_image_cropped = Data_image_cropped[0]

        Data_image_cropped = np.nan_to_num(Data_image_cropped)
        Data_image_cropped = cut_off_outliers(Data_image_cropped, percentile_lower=0.2, percentile_upper=99.8, per_channel=True)

        T2_mean += np.mean(Data_image_cropped[2,:,:,:])
        T2_std += np.std(Data_image_cropped[2,:,:,:])
        ADC_mean += np.mean(Data_image_cropped[0,:,:,:])
        ADC_std += np.std(Data_image_cropped[0,:,:,:])
        BVAL_mean += np.mean(Data_image_cropped[1,:,:,:])
        BVAL_std += np.std(Data_image_cropped[1,:,:,:])


        Tumor_frequencie_ADC += np.sum(Data_class_Label[0, :, :, :] == 2)
        Prostate_frequencie_ADC += np.sum(Data_class_Label[0, :, :, :] == 1)
        Background_frequencie_ADC += np.sum(Data_class_Label[0, :, :, :] == 0)

        Tumor_frequencie_T2 += np.sum(Data_class_Label[1, :, :, :] == 2)
        Prostate_frequencie_T2 += np.sum(Data_class_Label[1, :, :, :] == 1)
        Background_frequencie_T2 += np.sum(Data_class_Label[1, :, :, :] == 0)

    T2_mean = T2_mean / np.float(len(train_idx))
    T2_std = T2_std / np.float(len(train_idx))
    ADC_mean = ADC_mean / np.float(len(train_idx))
    ADC_std = ADC_std / np.float(len(train_idx))
    BVAL_mean = BVAL_mean / np.float(len(train_idx))
    BVAL_std = BVAL_std / np.float(len(train_idx))

    return Tumor_frequencie_ADC, Prostate_frequencie_ADC, Background_frequencie_ADC,\
           Tumor_frequencie_T2, Prostate_frequencie_T2, Background_frequencie_T2, ADC_mean, ADC_std, BVAL_mean, \
           BVAL_std, T2_mean, T2_std


def get_oversampling(Data, train_idx, Batch_Size, patch_size):

    IDs = sorted(train_idx)

    Total_Patients = len(IDs)

    print 'Total_Patients', Total_Patients

    Tumor_slices = 0
    Prostate_slices = 0
    Pos_Patient = 0
    Non_Tumor_Slices = 0
    Non_Prostate_Slices = 0

    for i in range(len(IDs)):
        idx = IDs[i]
        print(idx)
        z_Dim = (Data[idx]['label']).shape
        Data_over = Data[idx]['label']

        center_crop_dimensions = ((patch_size[0], patch_size[1]))
        Data_over_cropped = center_crop(Data_over, center_crop_dimensions)
        Data_over_cropped = Data_over_cropped[0]

        for f in range(z_Dim[3]):
            if np.sum(Data_over_cropped[0,:,:,f] == 2) >= 1:
                Tumor_slices += 1
            else:
                Non_Tumor_Slices += 1

            if np.sum(Data_over_cropped[0,:,:,f] == 1) >= 1:
                Prostate_slices += 1
            else:
                Non_Prostate_Slices += 1

        if np.sum(Data_over_cropped == 2) >= 1:
            Pos_Patient += 1


        print 'Tumor', Tumor_slices
        print 'Non_Tumor', Non_Tumor_Slices
        print 'Pos_Patients', Pos_Patient

        print 'Prostate', Prostate_slices
        print 'Non_Prostate_Slices', Non_Prostate_Slices



    print(Tumor_slices, Non_Tumor_Slices)


    Probability_Pos_Patient = Pos_Patient / np.float(Total_Patients)

    print 'Probability_Pos_Patient', Probability_Pos_Patient

    Slices_total = Tumor_slices + Non_Tumor_Slices

    Natural_probability_tu_slice = Tumor_slices/ np.float(Slices_total)

    Natural_probability_Prostate_slice = Prostate_slices / np.float(Slices_total)

    print 'Natural_probability_tu_slice', Natural_probability_tu_slice
    print 'Natural_probability_PRO_slice', Natural_probability_Prostate_slice

    Oversampling_Factor = (Natural_probability_tu_slice ** (1/np.float(Batch_Size * Probability_Pos_Patient)))

    print 'oversampling_factor', Oversampling_Factor

    return (Oversampling_Factor), Slices_total, Natural_probability_tu_slice, Natural_probability_Prostate_slice




def split_training(IDs, len_val, cv, cv_runs):

    runs = cv_runs / 4
    for s in range(runs):
        if cv in range(s * 4, s * 4 + 4):
            count = s
    print count

    np.random.seed(42 + count)

    IDx = np.array(IDs)
    np.random.shuffle(IDx)

    factor = cv - count * 4

    print factor
    lower_limit = len_val * factor
    print(lower_limit)

    upper_limit = len_val * (factor + 1)

    if factor == 3:
        upper_limit = len(IDx)
    print(upper_limit)

    val_idx = IDx[lower_limit:upper_limit]
    train_idx = np.setdiff1d(IDs, val_idx)

    return train_idx, val_idx


def save(Image, OutputFilePath, Mask = False):

    if Mask == True:
        sitk.WriteImage(sitk.Cast(Image, sitk.sitkUInt8), OutputFilePath)
    else:
        sitk.WriteImage(sitk.Cast(Image, sitk.sitkFloat32), OutputFilePath)

    finalImg = sitk.ReadImage(OutputFilePath)
    finalImg = sitk.GetArrayFromImage(finalImg)
    finalImg = finalImg.swapaxes(0, 2)
    finalImg_data, finalImg_options = nrrd.read(OutputFilePath)
    finalImg_options['encoding'] = 'gzip'
    nrrd.write(OutputFilePath, finalImg, header=finalImg_options)