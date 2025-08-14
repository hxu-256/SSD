import numpy as np
from scipy.spatial.distance import cosine
import scipy.io as sio
import cv2
from pytorch_msssim import ssim
import torch
import os
from skimage.filters import gaussian
from skimage.metrics import structural_similarity


def calculate_psnr(reference, data):
    temp_psnr = 0.
    reference = (reference - np.max(reference)) / (np.min(reference) - np.max(reference))
    data = (data - np.max(data)) / (np.min(data) - np.max(data))
    for i in range(reference.shape[2]):
        mse = np.mean((reference[:, :, i] - data[:, :, i])**2)
        if mse == 0:
            temp_psrn = 100
        Pixel_max = np.max(reference[:, :, i])
        if Pixel_max <= 0:
            temp_psnr += 0
        else:
            temp_psnr += 20 * np.log10(Pixel_max / np.sqrt(mse))
    return temp_psnr / reference.shape[2]

def calculate_ssim(reference, data):
    reference = (reference - np.max(reference)) / (np.min(reference) - np.max(reference))
    data = (data - np.max(data)) / (np.min(data) - np.max(data))
    return ssim(torch.unsqueeze(torch.from_numpy(reference).float(), 0).permute(3, 0, 1, 2), torch.unsqueeze(torch.from_numpy(data).float(), 0).permute(3, 0, 1, 2), data_range=1).data

def calculate_feature_sim(reference, data):
    reference = (reference - np.max(reference)) / (np.min(reference) - np.max(reference))
    data = (data - np.max(data)) / (np.min(data) - np.max(data))
    reference_features = gaussian(reference, sigma=1, multichannel=True)
    data_features = gaussian(data, sigma=1, multichannel=True)
    feature_sim_values = np.zeros(reference.shape[2])
    for i in range(reference.shape[2]):
        feature_sim_values[i] = structural_similarity(reference_features[:,:,i], data_features[:,:,i])
    return np.mean(feature_sim_values)

def calculate_sam(reference, data):
    reference = (reference - np.max(reference)) / (np.min(reference) - np.max(reference))
    data = (data - np.max(data)) / (np.min(data) - np.max(data))
    reference_flat = reference.reshape(-1, reference.shape[2])
    data_flat = data.reshape(-1, data.shape[2])
    sam_values = np.zeros(reference_flat.shape[0])
    for i in range(reference_flat.shape[0]):
        sam_values[i] = np.arccos(1 - cosine(reference_flat[i], data_flat[i]))
    return np.mean(sam_values)

def evaluate_all_results(reference, data):
    r_psnr = calculate_psnr(reference, data)
    r_ssim = calculate_ssim(reference, data)
    r_fsim = calculate_feature_sim(reference, data)
    r_sam = calculate_sam(reference, data)
    print(f"{r_psnr:.2f}")
    print(f"{r_ssim:.3f}")
    print(f"{r_fsim:.3f}")
    print(f"{r_sam:.3f}")
    print('\n')



if __name__ == '__main__':
    load_stage = False
    dataset_name = 'Simu'
    result_file_path = 'G:/Working_Yurong/Working_RamanDenoise/Ours/All_results/'
    ref_file_path = 'G:/Working_Yurong/Working_RamanDenoise/Raman_Denoising_Dataset/'
    idx = 1


    #------------------------- Load data -------------------------#
    if dataset_name == 'CMa':
        data_path = result_file_path + 'Cell_CMa/CMa' + str(idx)   
        reference_path = ref_file_path + '/Cell_CMa/CMa' + str(idx) + '.mat'
    elif dataset_name == 'Simu':
        data_list = ['Chessboard_data', 'Gaussian_data', 'Pattern_data']  
        data_path = result_file_path + '/Simu_data/' + data_list[idx-1]
        reference_path = ref_file_path + '/Simu_data/' + data_list[idx-1] + '.mat'
    elif dataset_name == 'Tablet':
        data_list = ['Tablet']  
        data_path = result_file_path + '/Tablet_data/'
        reference_path = ref_file_path + 'Tablet_data/' + data_list[idx-1] + '.mat'
        reference = sio.loadmat(reference_path)['HR_RHSI']
        reference = (reference - np.min(reference)) / (np.max(reference) - np.min(reference))
        noisy_rhsi = sio.loadmat(reference_path)['Noisy_RHSI']
        load_stage = True
    
    if load_stage == False:
        reference = sio.loadmat(reference_path)['HR_RHSI']
        noisy_rhsi = sio.loadmat(reference_path)['Noisy_RHSI']


    #------------------------ Test result ------------------------#
    print(20*'-', 'Noisy_RHSI')
    noisy_rhsi = (noisy_rhsi - np.min(noisy_rhsi)) / (np.max(noisy_rhsi) - np.min(noisy_rhsi))
    evaluate_all_results(reference, noisy_rhsi)

    for file_idx in os.listdir(data_path):
        mat_data = sio.loadmat(data_path + '/' + file_idx)
        print(20*'-', 'Method:', file_idx[0:-1], ' / Select data:', list(mat_data.keys())[-1])

        data = mat_data[list(mat_data.keys())[-1]]
        if np.max(data) > 10:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            data = data/255

        #data = data*np.max(noisy_rhsi)/2
        print(f"Max GT: {np.max(reference):.3f}", f"Min GT: {np.min(reference):.3f}", f"Max Data: {np.max(data):.3f}", f"Min Data: {np.min(data):.3f}")
        evaluate_all_results(reference, data)