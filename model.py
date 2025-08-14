##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                          Hunan University                            ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import os
import torch
import scipy.io as sio
from func import *
from Networks.Network import Network_load
import torch.nn.functional as F
from thop import profile
import  metric as final_metric
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def ADMM_Iter(GT_data_tensor, meas_data_tensor, result_dir, args):
    # -------------------- Initialization -------------------- #
    y0 = meas_data_tensor.squeeze().permute(1, 2, 0).to(device)
    z = y0
    u1, s = torch.zeros_like(z).to(device), torch.zeros_like(z).to(device)
    

    # ----------------------- Iteration ---------------------- #
    for it in range(args.iter_num):
        # -------- Updata x and s
        x = (y0 - s + args.rho*(z - u1)) / (1 + args.rho)
        s = shrink(y0 - x, args.lambda_S)
        #if args.vis_plot:
        #    cv2.imshow('Sparse noise', cv2.resize(abs(s[:, :, 455].cpu().numpy()*10), (400, 400)))
        #    cv2.waitKey(1)
        
        # -------- Updata z
        z = (x + u1).permute(2, 0, 1).unsqueeze(0)
        z = MSD_model(GT_data_tensor, z, meas_data_tensor, args)
        z = z.squeeze(0).permute(1, 2, 0).to(device)
        
        # -------- Updata Dual Variable u1
        u1 = u1 + (x - z)
        args.lambda_R = args.lambda_R*args.lambda_D
        
        # -------- Evaluation
        r_psnr = final_metric.calculate_psnr(GT_data_tensor.squeeze(0).permute(1,2,0).cpu().numpy(), z.cpu().numpy())
        r_ssim = r_psnr #final_metric.calculate_ssim(GT_data_tensor.squeeze(0).permute(1,2,0).cpu().numpy(), z.cpu().numpy())
        r_fsim = r_psnr #final_metric.calculate_feature_sim(GT_data_tensor.squeeze(0).permute(1,2,0).cpu().numpy(), z.cpu().numpy())
        r_sam = final_metric.calculate_sam(GT_data_tensor.squeeze(0).permute(1,2,0).cpu().numpy(), z.cpu().numpy())
        print('Iter {}'.format(it+1), f"Recon PSNR: {r_psnr:.3f}", f"Recon SSIM: {r_ssim:.3f}", f"Recon FSIM: {r_fsim:.3f}", f"Recon SAM: {r_sam:.3f}")
        sio.savemat(result_dir + 'Iter-{}_{:.3f}_{:.3f}_{:.3f}_{:.3f}.mat'.format(it, r_psnr, r_ssim, r_fsim, r_sam), {'R_hsi': z.detach().cpu().numpy()})
    
    if os.path.exists('Results/model_weights.pth'):
        os.remove('Results/model_weights.pth')
    return z
    


def MSD_model(GT_data_tensor, z, meas_data_tensor, args):
    # -------------------- Initialization -------------------- #
    torch.backends.cudnn.benchmark, PI_img = True, None
    best_loss = float('inf')
    N_Layer, B, _, _ = z.shape
    im_net = Network_load(B)
    loss_l1, loss_l2 = torch.nn.L1Loss().to(device), torch.nn.MSELoss().to(device)


    # ---------------------- Load weight --------------------- #
    if os.path.exists('Results/model_weights.pth'):
        im_net.load_state_dict(torch.load('Results/model_weights.pth'))
        args.Epoc_num = 300
        print('----------------------- Load model weights -----------------------', f'lambda_R: {args.lambda_R:.5f}')
 
    if args.vis_plot:
        #flops, model_size = profile(im_net[0], inputs = (meas_data_tensor, None, ))
        #print('------- FLOPs: {:.3f} G'.format(flops/1000**3), '------- Model Size: {:.3f} MB'.format(model_size/1024**2))
        plt.ion()
        fig, ax = plt.subplots(figsize=(11, 3))
        
    im_net.train()
    optimizer = torch.optim.Adam([{'params': list(im_net.parameters()), 'lr': 1e-3}])

    
    # ----------------------- Iteration ---------------------- #
    for idx in range(args.Epoc_num):
        model_out = im_net(z, PI_img)
        PI_img = torch.mean(model_out.detach(), 1).unsqueeze(1)

        # -------- Caculate loss
        loss = loss_l1(model_out, meas_data_tensor) #+ loss_l2(model_out, z)
        for i in range(N_Layer):
            loss_tv = args.lambda_R*calculate_tv(model_out[i, :, :, :].permute(1, 2, 0))
            loss_stv = args.lambda_STV*args.lambda_R*calculate_stv(model_out[i, :, :, :].permute(1, 2, 0))

        loss = loss + loss_tv + loss_stv
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        
        # -------- Save best results
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_hs_recon = model_out.detach()
            if args.PnP_i == True:
                torch.save(im_net.state_dict(), 'Results/model_weights.pth')

        if (idx+1)%100==0:
            SAM = final_metric.calculate_sam(GT_data_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), model_out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
            print('Iter {}, x_loss:{:.3f}, tv_loss:{:.3f}, stv_loss:{:.3f}, SAM:{:.3f}'.format(idx+1, loss.item()*10, loss_tv.item()*10000, loss_stv.item()*10000, SAM))
            
        # -------- Plot recon. result
        if args.vis_plot:
            model_out_show = (model_out - torch.min(model_out)) / (torch.max(model_out) - torch.min(model_out)) 
            GT_img  = GT_data_tensor[:, 480, :, :].squeeze().detach().cpu().numpy()*1.5
            Z_img = z[:, 480, :, :].squeeze().detach().cpu().numpy()*1.5
            Noisy_img = meas_data_tensor[:, 480, :, :].squeeze().detach().cpu().numpy()*1.5
            Recon_img = model_out_show[:, 480, :, :].squeeze().detach().cpu().numpy()*1.5
            cv2.imshow('Band', cv2.resize(abs(np.hstack((GT_img, Z_img, Noisy_img, Recon_img))), (900, 200)))
            cv2.waitKey(1)

            GT_spectrum = GT_data_tensor[0, :, 48, 63].detach().cpu().numpy()
            #Noisy_spectrum = meas_data_tensor[0, :, 48, 63].detach().cpu().numpy()
            Recon_spectrum = model_out_show[0, :, 48, 63].detach().cpu().numpy()
            ax.clear()
            ax.plot(GT_spectrum, label='GT Spectrum', linewidth=1.0)
            #ax.plot(Noisy_spectrum, label='Noisy Spectrum', linewidth=1.0)
            ax.plot(Recon_spectrum, label='Recon Spectrum', linewidth=1.2)
            ax.legend()
            plt.draw()
            plt.pause(0.1)

    plt.ioff() 
    plt.close() 
    return best_hs_recon