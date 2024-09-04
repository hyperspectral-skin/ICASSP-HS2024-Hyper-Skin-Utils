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
parser.add_argument('--model_name', type = str, default = 'mstpp', required=True, help = 'the model name')
parser.add_argument('--models_dir', type = str, default = 'models', required=False, help = 'directory of the trained model')
parser.add_argument('--results_dir', type = str, default = 'results', required=False, help = 'to save the evaluation results')
parser.add_argument('--reconstructed_dir', type = str, default = 'reconstructed-hsi', required=False, help = 'directory to save the reconstructed HSI')


if __name__ == '__main__':
    args = parser.parse_args()

    #################################################################
    # directory to the MSI (RGB + 960) and VIS, NIR data  
    test_msi_dir = f'{args.data_dir}/Hyper-Skin(MSI, NIR)/test/MSI_CIE'
    test_nir_dir = f'{args.data_dir}/Hyper-Skin(MSI, NIR)/test/NIR'
    test_vis_dir = f'{args.data_dir}/Hyper-Skin(RGB, VIS)/test/VIS'
    mask_dir = f'{args.data_dir}/Hyper-Skin(MSI, NIR)/mask/test'
    masks = sorted(os.listdir(mask_dir))

    #################################################################
    # define the test datasets  
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



    model = torchmodel.create(
            model_architecture = model_architecture,
            device=device,
            loss_fn=None,
            optimizer=None,
            scheduler=None,
            epochs = config.epochs,
            logger=None,
            model_saved_path=f'{args.models_dir}/mstpp.pt'
    )


    # evaluation
    model.load_model(model_architecture)
    model.model.to(device)
    model.model.eval() 
    results = {
        "file": [],
        "pred": [],
        "sam_score": [],
        "sam_map": []
    }

    for k, data in enumerate(test_loader):
        x, y = data 
        x, y = x.float().to(device), y.float().to(device)

        mask = plt.imread(f"{mask_dir}/{masks[k]}")/255
        mask = torch.from_numpy(mask[:,:,0]).float().to(device)

        with torch.no_grad():
            pred = model.model(x)
            pred = pred * mask[None, None, ...]
            y = y * mask[None, None, ...]

        sam_score, sam_map = utils.sam_fn(pred, y)

        results["file"].append(test_dataset.msi_files[k].split('\\')[-1].split('.')[0])
        results["pred"].append(pred.cpu().detach().numpy())
        results["sam_score"].append(sam_score.cpu().detach().numpy())
        results["sam_map"].append(sam_map.cpu().detach().numpy())
        
        print(f"Test [{k+1}/{len(test_loader)}]: {results['file'][-1]}, SAM: {results['sam_score'][-1]}")    

    sam_scores = np.array(results["sam_score"])
    sam_mean = np.mean(sam_scores)
    sam_std = np.std(sam_scores)
    print(f"SAM: {sam_mean:.4f} +/- {sam_std:.4f}")
    #################################################################
    # save the results into the pickle files
    with open(f'{args.results_dir}/{args.model_name}.pkl', 'wb') as f:
        pickle.dump(results, f)

    # select a random image to visualize
    k = 4  
    utils.visualize_save_cube63(
        cube = results["pred"][k].squeeze(), 
        bands = np.arange(400, 1010, 10),
        rgb_file = test_dataset.msi_files[k], MSI = True,
        saved_path = f"{args.results_dir}/test-{results['file'][k]}-{args.model_name}")
    
    # visualize the sam map
    utils.visualize_sam_map(
        np.array(results["sam_map"]).squeeze(), 
        saved_path = f"{args.results_dir}/test-sam-map-{args.model_name}")

