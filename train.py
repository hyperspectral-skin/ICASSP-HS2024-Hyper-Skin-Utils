import os 
import pickle
import argparse
import numpy as np 
import matplotlib.pyplot as plt 

import config
import utils, torchmodel
from hsiData import HyperSkinData
from baseline import MST_Plus_Plus  #, HSCNN_Plus, HDNet, hrnet

import torch
import torchinfo
from torchvision import transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = 'D:/ICASSP2024-SPGC', required=True, help = 'data directory')
parser.add_argument('--model_name', type = str, default = 'mstpp', required=True, help = 'the model')
parser.add_argument('--saved_dir', type = str, default = 'models', required=False, help = 'directory to save the trained model')
parser.add_argument('--logged_dir', type = str, default = 'logs', required=False, help = 'directory to save the results')
parser.add_argument('--external_dir', type = str, default = None, required=False, help = 'create and save the log, reconstruction and trained model to external directory')


if __name__ == '__main__':
    args = parser.parse_args()

    #################################################################
    # directory to the MSI (RGB + 960) and NIR data
    train_msi_dir = f'{args.data_dir}/Hyper-Skin(MSI, NIR)/train/MSI_CIE'
    train_nir_dir = f'{args.data_dir}/Hyper-Skin(MSI, NIR)/train/NIR'
    train_vis_dir = f'{args.data_dir}/Hyper-Skin(RGB, VIS)/train/VIS'
    
    test_msi_dir = f'{args.data_dir}/Hyper-Skin(MSI, NIR)/test/MSI_CIE'
    test_nir_dir = f'{args.data_dir}/Hyper-Skin(MSI, NIR)/test/NIR'
    test_vis_dir = f'{args.data_dir}/Hyper-Skin(RGB, VIS)/test/VIS'

    # create the saved  and log folders if they not exist
    exp_logged_dir, exp_saved_dir = utils.create_folders_for(
                                                saved_dir = args.saved_dir, 
                                                logged_dir = args.logged_dir, 
                                                model_name = args.model_name,
                                                external_dir = args.external_dir)
    logger = utils.initiate_logger(f'{exp_logged_dir}/log', 'train')
    #################################################################


    #################################################################
    # define the train, valid and test datasets  - use v2 for nir and msi data
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(config.crop_size, config.crop_size)
    ])
    train_dataset = HyperSkinData.Load_msi_visnir(
            msi_dir = train_msi_dir,
            vis_dir = train_vis_dir, 
            nir_dir = train_nir_dir,
            transform = train_transform,
    )

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = HyperSkinData.Load_msi_visnir(
            msi_dir = test_msi_dir,
            vis_dir = test_vis_dir,
            nir_dir = test_nir_dir,
            transform = test_transform,
    )

    # define the dataloaders
    train_loader = torch.utils.data.DataLoader(
                                    dataset = train_dataset, 
                                    batch_size = config.batch_size, 
                                    shuffle = True, 
                                    pin_memory = True, 
                                    drop_last = True)
    
    test_loader = torch.utils.data.DataLoader(
                                    dataset = test_dataset, 
                                    batch_size = 1, 
                                    shuffle = False, 
                                    pin_memory = False)
    #################################################################


    #################################################################
    # define the model
    if args.model_name == 'mstpp':
        model_architecture = MST_Plus_Plus.MST_Plus_Plus(in_channels=4, out_channels=61, n_feat=61, stage=3)
    # elif args.model_name == 'HSCNN':
    #     model_architecture = HSCNN_Plus.HSCNN_Plus(in_channels=4, out_channels=31, num_blocks=20)
    # elif args.model_name == 'HRNET':
    #     model_architecture = hrnet.SGN(in_channels=4, out_channels=31)
    # elif args.model_name == 'HDNET':
    #     model_architecture = HDNet.HDNet()
    total_model_parameters = sum(p.numel() for p in model_architecture.parameters())

    msg = f"[Experiment Metadata]:\n" +\
            f"Model: {args.model_name}\n" +\
            f"Total Model Parameters: {total_model_parameters}\n" +\
            f"Trained Model will be saved at: {exp_saved_dir}\n" +\
            f"Log file available at: {exp_logged_dir}\n"+\
            f"data directory (MSI and NIR): {train_msi_dir}, {train_nir_dir}\n"+\
            "==================================================================================================================="
    logger.info(msg)
    print(msg)

    optimizer = torch.optim.Adam(model_architecture.parameters(), lr=config.init_lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=1e-6)
    loss_fn = torch.nn.L1Loss()

    model = torchmodel.create(
            model_architecture = model_architecture,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs = config.epochs,
            logger=logger,
            model_saved_path=f'{exp_saved_dir}/{args.model_name}.pt'
    )

    # training
    best_valid_loss, history = model.train(
        train_loader = train_loader,
        valid_loader = test_loader,
        best_valid_loss = 100,
    )

    # # evaluation
    # model.load_model(model_architecture)
    # model.model.to(device)
    # model.model.eval() 
    # results = {
    #     "file": [],
    #     "pred": [],
    #     "ssim_score": [],
    #     "ssim_map": [],
    #     "sam_score": [],
    #     "sam_map": []
    # }
    # utils.ssim_fn.to(device)
    # for k, data in enumerate(test_loader):
    #     x, y = data 
    #     x, y = x.float().to(device), y.float().to(device)

    #     with torch.no_grad():
    #         pred = model.model(x)

    #     ssim_score, ssim_map = utils.ssim_fn(pred, y)
    #     sam_score, sam_map = utils.sam_fn(pred, y)

    #     results["file"].append(test_dataset.rgb_files[k].split('\\')[-1].split('.')[0])
    #     results["pred"].append(pred.cpu().detach().numpy())
    #     results["ssim_score"].append(ssim_score.cpu().detach().numpy())
    #     results["ssim_map"].append(ssim_map.cpu().detach().numpy())
    #     results["sam_score"].append(sam_score.cpu().detach().numpy())
    #     results["sam_map"].append(sam_map.cpu().detach().numpy())
        
    #     print(f"Test [{k}/{len(test_loader)}]: {results['file'][-1]}, SSIM: {results['ssim_score'][-1]}, SAM: {results['sam_score'][-1]}")    


    # #################################################################
    # # save into the pickle files
    # with open(f'{exp_logged_dir}/history.pkl', 'wb') as f:
    #     pickle.dump(history, f)

    # with open(f'{exp_logged_dir}/results.pkl', 'wb') as f:
    #     pickle.dump(results, f)

    # # save the reconstructed HSI in .mat format
    # if args.saved_predicted:
    #     for k, cube in enumerate(results["pred"]):
    #         filename = test_dataset.rgb_files[k].split('\\')[-1].split('.')[0]
    #         utils.save_matv73(f"{exp_reconstructed_dir}/{filename}.mat", 'cube', cube)

    #         # save the visualizations at all bands for each test image
    #         utils.visualize_save_cube32(
    #             cube = cube.squeeze(), 
    #             bands = config.bands, 
    #             rgb_file = test_dataset.rgb_files[k], MSI = True,
    #             saved_path = f"{exp_logged_dir}/{filename}.png")
