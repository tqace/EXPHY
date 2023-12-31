import torchvision.models as models
from PIL import Image
import torchvision.transforms.functional as TF
import imageio
import pandas as pd
import argparse
import torchvision
import torch
import os
from src.exphy import IODINE
from src.networks.refine_net import RefineNetLSTM
from src.networks.sbd import SBD
from src.datasets.datasets import ComphyDataset

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_args_parser():
    parser = argparse.ArgumentParser(description='Set hyperparameters for the model.')
    parser.add_argument('--save_path', type=str, default='results', help='Path to save the trained models')
    parser.add_argument('--scenario', type=str, choices=['collision','charge'], help='Collision or Coulomb interaction')
    parser.add_argument('--pretrained_path', type=str, default='pretrained.th', help='Path to pretrained model')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Workers')
    parser.add_argument('--max_num_frames', type=int, default=6, help='Video length')
    parser.add_argument('--max_num_samples', type=int, default=50000, help='Dataset length')
    parser.add_argument('--down_sz', type=int, default=64, help='Image height')
    parser.add_argument('--n_pred_steps', type=int, default=5, help='Predicting steps')
    parser.add_argument('--T', type=int, default=5, help='Refine iterations')
    parser.add_argument('--K', type=int, default=8, help='Number of slots')
    parser.add_argument('--a_dim', type=int, default=12, help='Appearance code dim')
    parser.add_argument('--v_dim', type=int, default=2, help='Velocity code dim')
    parser.add_argument('--mc_dim', type=int, default=2, help='mass & charge code dim')
    parser.add_argument('--out_channels', type=int, default=4, help='Decoder output')
    parser.add_argument('--img_dim', type=str, default="64,96", help='Resolution')
    parser.add_argument('--beta', type=float, default=100., help='KL weight')
    parser.add_argument('--use_feature_extractor', action='store_true', help='Whether to use the feature extractor or not')

    return parser

def main(args):
    save_path = os.path.join(args.save_path,args.scenario)
    datapath = 'data/col' if args.scenario=='collision' else 'data/chg'
    pretrained_path = args.pretrained_path
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_workers
    max_num_frames = args.max_num_frames
    max_num_samples = args.max_num_samples
    down_sz = args.down_sz 
    n_pred_steps = args.n_pred_steps
    T = args.T 
    K = args.K
    a_dim = args.a_dim 
    v_dim = args.v_dim 
    mc_dim = args.mc_dim
    out_channels = args.out_channels 
    img_dim = tuple(map(int, args.img_dim.split(','))) 
    beta = args.beta 
    use_feature_extractor = True
    
    print('----------Scenario: ',args.scenario,'----------\n')

    #DataLoader
    dataloader = torch.utils.data.DataLoader(
	    ComphyDataset(datapath,max_num_samples=max_num_samples,max_num_frames=max_num_frames,down_sz=down_sz),
	    batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    #Model
    feature_extractor = models.squeezenet1_1(pretrained=True).features[:5] if use_feature_extractor else None
    refine_net_a = RefineNetLSTM(a_dim,32)
    refine_net_v = RefineNetLSTM(v_dim,16)
    refine_net_mc = RefineNetLSTM(mc_dim,58)
    decoder = SBD(a_dim,img_dim,out_channels=out_channels)
    v = IODINE(n_pred_steps,refine_net_a,refine_net_v,refine_net_mc,decoder,T,K,a_dim,v_dim,mc_dim,
	    feature_extractor=feature_extractor,beta=beta)
    v.load(pretrained_path)
    v = v.to(device)
    v.eval()
            

    for i,mbatch in enumerate(dataloader):
        x = mbatch.to(device)[:,:]		
        gt = x
        N,F,C,H,W = x.shape
        observation_frames = []
        explain_frames = [] 
        counterfactual1_frames = []  
        counterfactual2_frames = []  
        new_size = (96*2, 64*2)
        #Observation
        for j in range(6):
            gt_img = x[0][j]
            gt_img = TF.to_pil_image(gt_img)
            gt_img = gt_img.resize(new_size, Image.BILINEAR)
            observation_frames.append(gt_img)
            if j==0:
                explain_frames.append(gt_img)
                counterfactual1_frames.append(gt_img)
                counterfactual2_frames.append(gt_img)
        #Results
        explain_ret,counterfactual_ret1,counterfactual_ret2 = v.interpret(x,args.scenario)	
        
        #Explanation
        for j in range(5):
            explain_img = explain_ret['image'][j]
            explain_img = TF.to_pil_image(explain_img)  
            explain_img = explain_img.resize(new_size, Image.BILINEAR)  
            explain_frames.append(explain_img) 
        del explain_ret['image']
        df = pd.DataFrame(explain_ret)
        df_string = df.to_string(index=False)
        print('Explanation \n',df_string,'\n \n')
        #Counterfactual 1
        for j in range(5):
            counterfactual1_img = counterfactual_ret1['image'][j]
            counterfactual1_img = TF.to_pil_image(counterfactual1_img)
            counterfactual1_img = counterfactual1_img.resize(new_size, Image.BILINEAR)
            counterfactual1_frames.append(counterfactual1_img)
        del counterfactual_ret1['image']
        df = pd.DataFrame(counterfactual_ret1)
        df_string = df.to_string(index=False)
        print('Counterfactual 1 \n',df_string,'\n \n')
        #Counterfactual 2
        for j in range(5):
            counterfactual2_img = counterfactual_ret2['image'][j]
            counterfactual2_img = TF.to_pil_image(counterfactual2_img)
            counterfactual2_img = counterfactual2_img.resize(new_size, Image.BILINEAR)
            counterfactual2_frames.append(counterfactual2_img)
        del counterfactual_ret2['image']
        df = pd.DataFrame(counterfactual_ret2)
        df_string = df.to_string(index=False)
        print('Counterfactual 2 \n',df_string)
        
        imageio.mimsave(save_path + '/observation.gif', observation_frames, duration=0.5)

        imageio.mimsave(save_path + '/explain.gif', explain_frames, duration=0.5)

        imageio.mimsave(save_path +'/counterfactual_1.gif', counterfactual1_frames, duration=0.5)

        imageio.mimsave(save_path + '/counterfactual_2.gif', counterfactual2_frames, duration=0.5)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
