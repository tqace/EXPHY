import torch
import torch.nn.init as init
import random
import scipy.optimize
from torch import nn
import math
from .utils.util import adjusted_rand_index, adjusted_rand_index_without_bg
import os
import numpy as np
"""
Implementation of  Iterative Object Decomposition Inference Network (IODINE)
from "Multi-Object Representation Learning with Iterative Variational Inference" by Greff et. al. 2019
Link: https://arxiv.org/pdf/1903.00450.pdf
"""
import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
class IODINE(torch.nn.Module):

    def __init__(self,
            n_pred_steps,
            refine_net,
            refine_net_dyn,
            refine_net_mc,
            decoder,
            T,
            K,
            a_dim,
            v_dim,
            mc_dim,
            beta=1.,
            feature_extractor=None):
        super(IODINE, self).__init__()
        self.n_pred_steps = n_pred_steps
        self.lmbda0 = torch.nn.Parameter(torch.rand(1,2*a_dim)-0.5,requires_grad=True)
        self.lmbda1 = torch.nn.Parameter(torch.rand(1,2*v_dim)-0.5,requires_grad=True) 
        self.lmbda2 = torch.nn.Parameter(torch.rand(1,2*mc_dim)-0.5,requires_grad=True)
        self.decoder = decoder
        self.refine_net = refine_net
        self.refine_net_dyn = refine_net_dyn
        self.refine_net_mc = refine_net_mc
        self.layer_norms = torch.nn.ModuleList([
                torch.nn.LayerNorm((1,64,96),elementwise_affine=False),
                torch.nn.LayerNorm((3,64,96),elementwise_affine=False),
                torch.nn.LayerNorm((1,64,96),elementwise_affine=False),
                torch.nn.LayerNorm((2*a_dim,),elementwise_affine=False),
                torch.nn.LayerNorm((2*a_dim,),elementwise_affine=False),

                torch.nn.LayerNorm((1,64,96),elementwise_affine=False),
                torch.nn.LayerNorm((3,64,96),elementwise_affine=False),
                torch.nn.LayerNorm((1,64,96),elementwise_affine=False),
                torch.nn.LayerNorm((2*v_dim,),elementwise_affine=False), 
                torch.nn.LayerNorm((2*mc_dim,),elementwise_affine=False),
                torch.nn.LayerNorm((1,64,96),elementwise_affine=False)
                ])
        self.use_feature_extractor = feature_extractor is not None
        if self.use_feature_extractor:
                self.feature_extractor = torch.nn.Sequential(
                        feature_extractor,
                        torch.nn.Conv2d(128,64,3,stride=1,padding=1),
                        torch.nn.ELU(),
                        torch.nn.Conv2d(64,32,3,stride=1,padding=1),
                        torch.nn.ELU(),
                        torch.nn.Conv2d(32,16,3,stride=1,padding=1),
                        torch.nn.ELU())
                for param in self.feature_extractor[0]:
                        param.requires_grad = False
        
        self.register_buffer('T', torch.tensor(T))
        self.register_buffer('K', torch.tensor(K))
        self.register_buffer('a_dim', torch.tensor(a_dim))
        self.register_buffer('var_x', torch.tensor(0.3))
        self.register_buffer('h0',torch.zeros((1,128)))
        self.register_buffer('base_loss',torch.zeros(1,1))
        self.register_buffer('b', torch.tensor(beta)) 
        self.interactor_pre = Interactor()
        self.interactor = nn.Linear(2, a_dim,bias=False)

    def interpret(self, img, scenario):
                img = img.permute(1,0,2,3,4)
                        
                F,N,C,H,W = img.shape
                K, T, a_dim = self.K, self.T, self.a_dim
                assert not torch.isnan(self.lmbda0).any().item(), 'lmbda0 has nan'
                
                ## Initialize parameters for latents' distribution
                lmbda_frames = self.lmbda0.expand((F-self.n_pred_steps, N*K,)+self.lmbda0.shape[1:])
                lmbda_dyn_frames = self.lmbda1.expand((F-self.n_pred_steps, N*K,)+self.lmbda1.shape[1:])
                lmbda_mc_frames = self.lmbda2.expand((F-self.n_pred_steps, N*K,)+self.lmbda2.shape[1:])
                total_loss = torch.zeros_like(self.base_loss.expand((N,1)))
                for f in range(F-self.n_pred_steps):
                    x = img[f]
                    lmbda = lmbda_frames[f]
                    lmbda_dyn = lmbda_dyn_frames[f]
                    lmbda_mc = lmbda_mc_frames[f]
                    ## Initialize LSTMCell hidden states
                    h = self.h0.expand((N*K,)+self.h0.shape[1:]).clone().detach() 
                    c = torch.zeros_like(h)
                    h_dyn = self.h0.expand((N*K,)+self.h0.shape[1:]).clone().detach() 
                    c_dyn = torch.zeros_like(h_dyn)
                    h_mc = self.h0.expand((N*K,)+self.h0.shape[1:]).clone().detach() 
                    c_mc = torch.zeros_like(h_mc)
                    assert h.max().item()==0. and h.min().item()==0.

                    for it in range(T):
                        _x_all = []
                        mu_x_all = []
                        masks_all = []
                        mask_logits_all = []
                        ll_pxl_all = []
                        deviation_all = []
                        # stage reconstruction    
                        ## Sample latent code
                        mu_z, logvar_z = lmbda.chunk(2,dim=1)
                        mu_z, logvar_z = mu_z.contiguous(), logvar_z.contiguous()
                        z = self._sample(mu_z,logvar_z) ## (N*K,z_dim)
                        mu_z_dyn, logvar_z_dyn = lmbda_dyn.chunk(2,dim=1)
                        mu_z_dyn, logvar_z_dyn = mu_z_dyn.contiguous(), logvar_z_dyn.contiguous()
                        z_dyn = self._sample(mu_z_dyn,logvar_z_dyn) ## (N*K,z_dim)
                        mu_z_mc, logvar_z_mc = lmbda_mc.chunk(2,dim=1)
                        mu_z_mc, logvar_z_mc = mu_z_mc.contiguous(), logvar_z_mc.contiguous()
                        z_mc = self._sample(mu_z_mc,logvar_z_mc) ## (N*K,z_dim)
                        
                        ## Get means and masks 
                        dec_out = self.decoder(z) ## (N*K,C+1,H,W)
                        mu_x, mask_logits = dec_out[:,:C,:,:], dec_out[:,C,:,:] ## (N*K,C,H,W), (N*K,H,W)
                        mask_logits = mask_logits.view((N,K,)+mask_logits.shape[1:]) ## (N,K,H,W)
                        mu_x = mu_x.view((N,K,)+mu_x.shape[1:]) ##(N,K,C,H,W)
                        mu_x_all.append(mu_x)

                        ## Process masks
                        masks = torch.nn.functional.softmax(mask_logits,dim=1).unsqueeze(dim=2) ##(N,K,1,H,W)
                        masks_all.append(masks)
                        mask_logits = mask_logits.unsqueeze(dim=2) ##(N,K,1,H,W)
                        mask_logits_all.append(mask_logits)
                        ## Calculate loss: reconstruction (nll) & KL divergence
                        _x = x.unsqueeze(dim=1).expand((N,K,)+x.shape[1:]) ## (N,K,C,H,W)
                        _x_all.append(_x)
                        deviation = -1.*(mu_x - _x)**2
                        deviation_all.append(deviation)
                        ll_pxl_channels = ((masks*(deviation/(2.*self.var_x)).exp()).sum(dim=1,keepdim=True)).log()
                        assert ll_pxl_channels.min().item()>-math.inf
                        ll_pxl = ll_pxl_channels.sum(dim=2,keepdim=True) ## (N,1,1,H,W)
                        ll_pxl_all.append(ll_pxl)
                        ll_pxl_flat = ll_pxl.view(N,-1)
                        nll = -1.*(ll_pxl_flat.sum(dim=-1).mean())
                        div = self._get_div(mu_z,logvar_z,N,K)
                        div_dyn = self._get_div(mu_z_dyn,logvar_z_dyn,N,K)
                        loss = self.b * nll + div + div_dyn
                        loss_pred = 0
                        loss_mc = 0
                        for i in range(self.n_pred_steps):
                            if i>0:
                                z_dyn_delta,_,_ = self.interactor_pre(z_dyn,z,z_mc)
                                z_dyn = z_dyn + z_dyn_delta
                            z_update = self.interactor(z_dyn)
                            z+=z_update
                            x_pred = img[f+i+1]
                            
                            ## Get means and masks 
                            dec_out_pred = self.decoder(z) ## (N*K,C+1,H,W)
                            mu_x_pred, mask_logits_pred = dec_out_pred[:,:C,:,:], dec_out_pred[:,C,:,:] ## (N*K,C,H,W), (N*K,H,W)
                            mask_logits_pred = mask_logits_pred.view((N,K,)+mask_logits_pred.shape[1:]) ## (N,K,H,W)
                            mu_x_pred = mu_x_pred.view((N,K,)+mu_x_pred.shape[1:]) ##(N,K,C,H,W)
                            mu_x_all.append(mu_x_pred)

                            ## Process masks
                            masks_pred = torch.nn.functional.softmax(mask_logits_pred,dim=1).unsqueeze(dim=2) ##(N,K,1,H,W)
                            masks_all.append(masks_pred)
                            mask_logits_pred = mask_logits_pred.unsqueeze(dim=2) ##(N,K,1,H,W)
                            mask_logits_all.append(mask_logits_pred)

                            ## Calculate loss: reconstruction (nll) & KL divergence
                            _x_pred = x_pred.unsqueeze(dim=1).expand((N,K,)+x_pred.shape[1:]) ## (N,K,C,H,W)
                            _x_all.append(_x_pred)
                            deviation_pred = -1.*(mu_x_pred - _x_pred)**2
                            deviation_all.append(deviation_pred)
                            ll_pxl_channels_pred = ((masks_pred*(deviation_pred/(2.*self.var_x)).exp()).sum(dim=1,keepdim=True)).log()
                            assert ll_pxl_channels_pred.min().item()>-math.inf
                            ll_pxl_pred = ll_pxl_channels_pred.sum(dim=2,keepdim=True) ## (N,1,1,H,W)
                            ll_pxl_all.append(ll_pxl_pred)
                            ll_pxl_flat_pred = ll_pxl_pred.view(N,-1)
                            
                            nll_pred = -1.*(ll_pxl_flat_pred.sum(dim=-1).mean())
                            loss_pred += self.b * nll_pred 
                            if i==0:
                                loss_dyn = loss_pred 
                            else:
                                loss_mc += self.b * nll_pred

                        ## Accumulate loss
                        scaled_loss = ((float(it)+1)/float(T)) * (loss + loss_pred)
                        total_loss += scaled_loss
                        
                        assert not torch.isnan(loss).any().item(), 'Loss at t={} is nan. (nll,div): ({},{})'.format(nll,div)
                        if it==T-1: continue

                        ## Refine lambda
                        refine_inp_rec = self.get_refine_inputs([_x_all[0]],[mu_x_all[0]],[masks_all[0]],[mask_logits_all[0]],[ll_pxl_all[0]],lmbda,loss,[deviation_all[0]],norm=3)
                        refine_inp_pred = self.get_refine_inputs([_x_all[1]],[mu_x_all[1]],[masks_all[1]],[mask_logits_all[1]],[ll_pxl_all[1]],lmbda_dyn,loss_dyn,[deviation_all[1]],norm=8)
                        refine_inp_mc = self.get_refine_inputs(_x_all[2:],mu_x_all[2:],masks_all[2:],mask_logits_all[2:],ll_pxl_all[2:],lmbda_mc,loss_mc,deviation_all[2:],norm=9)
        
                        ## Potentially add additional features from pretrained model (scaled down to appropriate size)
                        if self.use_feature_extractor:
                                x_resized = torch.nn.functional.interpolate(x,(257,385)) ## Upscale to desired input size for squeezenet
                                additional_features = self.feature_extractor(x_resized).unsqueeze(dim=1)
                                additional_features = additional_features.expand((N,K,16,64,96)).contiguous()
                                additional_features = additional_features.view((N*K,16,64,96))
                                refine_inp_rec['img'] = torch.cat((refine_inp_rec['img'],additional_features),dim=1)

                        delta, h, c = self.refine_net(refine_inp_rec, h, c)
                        delta_dyn, h_dyn, c_dyn = self.refine_net_dyn(refine_inp_pred, h_dyn, c_dyn)
                        delta_mc, h_mc, c_mc = self.refine_net_mc(refine_inp_mc, h_mc, c_mc)
                        lmbda = lmbda + delta
                        lmbda_dyn = lmbda_dyn + delta_dyn
                        lmbda_mc = lmbda_mc + delta_mc

                    ##explain & counterfactual
                    mu_z, logvar_z = lmbda.chunk(2,dim=1)
                    mu_z, logvar_z = mu_z.contiguous(), logvar_z.contiguous()
                    mu_z_dyn, logvar_z_dyn = lmbda_dyn.chunk(2,dim=1)
                    mu_z_dyn, logvar_z_dyn = mu_z_dyn.contiguous(), logvar_z_dyn.contiguous()
                    mu_z_mc, logvar_z_mc = lmbda_mc.chunk(2,dim=1)
                    mu_z_mc, logvar_z_mc = mu_z_mc.contiguous(), logvar_z_mc.contiguous()
                    z_mc = self._sample(mu_z_mc,logvar_z_mc) ## (N*K,z_dim)
                    z_dyn_ = self._sample(mu_z_dyn,logvar_z_dyn) ## (N*K,z_dim)
                    z_ = self._sample(mu_z,logvar_z) ## (N*K,z_dim)
                    k1,k2 = self.get_object_index(z_,z_dyn_,N,C,K)
                    if scenario=='collision':
                        if z_mc[k1,0] < z_mc[k2,0]:
                            tmp = k1
                            k1 = k2
                            k2 = tmp
                    else:
                        if z_mc[k1,1] < z_mc[k2,1]:
                            tmp = k1
                            k1 = k2
                            k2 = tmp
                    rets = []
                    for i in range(3):
                        z = z_.clone()
                        z_dyn = z_dyn_.clone()
                        final_masks_pred = []
                        final_mu_x_pred = []
                        ret = {'Frame':[],'Velocity':[],'Collision Acceleration':[],'Coulomb Acceleration':[]}
                        if i==1:
                            if scenario=='collision':
                                z_mc[k2,0] -= 4
                            else:
                                z_mc[k2,1] = z_mc[k1,1].clone()
                        if i==2:
                            if scenario=='collision':
                                z_mc[k2,0] += 4
                                z_mc[k1,0]-=4
                            else:
                                z_mc[k1,1]+=1.5
                                z_mc[k2,1]+=1.5

                        for j in range(self.n_pred_steps+1):
                            if j>0:
                                z_dyn_update,f_col,f_chg = self.interactor_pre(z_dyn,z,z_mc)
                                ret['Frame'].append(j)
                                ret['Velocity'].append((z_dyn.clone()[k1].cpu().detach().numpy(),z_dyn.clone()[k2].cpu().detach().numpy()))
                                ret['Collision Acceleration'].append((f_col[k1].cpu().detach().numpy(),f_col[k2].cpu().detach().numpy()))
                                ret['Coulomb Acceleration'].append((f_chg[k1].cpu().detach().numpy(),f_chg[k2].cpu().detach().numpy()))
                                z_dyn += z_dyn_update
                            z_update = self.interactor(z_dyn)
                            z+=z_update
                            ## Get means and masks 
                            dec_out_pred = self.decoder(z) ## (N*K,C+1,H,W)
                            mu_x_pred, mask_logits_pred = dec_out_pred[:,:C,:,:], dec_out_pred[:,C,:,:] ## (N*K,C,H,W), (N*K,H,W)
                            mask_logits_pred = mask_logits_pred.view((N,K,)+mask_logits_pred.shape[1:]) ## (N,K,H,W)
                            mu_x_pred = mu_x_pred.view((N,K,)+mu_x_pred.shape[1:]) ##(N,K,C,H,W)

                            ## Process masks
                            masks_pred = torch.nn.functional.softmax(mask_logits_pred,dim=1).unsqueeze(dim=2) ##(N,K,1,H,W)
                            mask_logits_pred = mask_logits_pred.unsqueeze(dim=2) ##(N,K,1,H,W)

                            final_mu_x_pred.append(mu_x_pred)
                            final_masks_pred.append(masks_pred)

                        output_means_pred = (torch.stack(final_mu_x_pred).permute(1,0,2,3,4,5) * torch.stack(final_masks_pred).permute(1,0,2,3,4,5)).sum(dim=2)
                        ret['image']=output_means_pred.squeeze(0)
                        rets.append(ret)

                return rets[0],rets[1],rets[2]
    
    def get_object_index(self,z,z_dyn,N,C,K):
        def decode(z):
            dec_out_pred = self.decoder(z) ## (N*K,C+1,H,W)
            mu_x_pred, mask_logits_pred = dec_out_pred[:,:C,:,:], dec_out_pred[:,C,:,:] ## (N*K,C,H,W), (N*K,H,W)
            mask_logits_pred = mask_logits_pred.view((N,K,)+mask_logits_pred.shape[1:]) ## (N,K,H,W)
            mu_x_pred = mu_x_pred.view((N,K,)+mu_x_pred.shape[1:]) ##(N,K,C,H,W)
            masks_pred = torch.nn.functional.softmax(mask_logits_pred,dim=1).unsqueeze(dim=2) ##(N,K,1,H,W)
            image = (mu_x_pred * masks_pred).sum(dim=1)
            return image
        z_update = self.interactor(z_dyn)
        img_org = decode(z+z_update)
        mses = []
        for i in range(self.K):
            z_dyn_cl = z_dyn.clone()
            z_dyn_cl[i]+=2
            z_update = self.interactor(z_dyn_cl)
            img = decode(z+z_update)
            mses.append(nn.MSELoss()(img,img_org))
        mses = torch.cat([mse.unsqueeze(0) for mse in mses])
        ranks =  torch.sort(mses,descending=True)[1]
        return ranks[0],ranks[1]

    def get_refine_inputs(self, _x_all,mu_x_all,masks_all,mask_logits_all,ll_pxl_all,lmbda,loss,deviation_all,norm):
        N,K,C,H,W = mu_x_all[0].shape
        
        ## Calculate additional non-gradient inputs
        ll_pxl_all = [ll_pxl.expand((N,K,) + ll_pxl.shape[2:]) for ll_pxl in ll_pxl_all] 
        p_mask_individual_all =[(deviation/(2.*self.var_x)).exp().prod(dim=2,keepdim=True) for deviation in deviation_all] 
        p_masks_all = [torch.nn.functional.softmax(p_mask_individual, dim=1) for p_mask_individual in p_mask_individual_all] 
        
        ## Calculate gradient inputs
        dmu_x_all = [torch.autograd.grad(loss,mu_x,retain_graph=True,only_inputs=True)[0] for mu_x in mu_x_all]
        dmasks_all = [torch.autograd.grad(loss,masks,retain_graph=True,only_inputs=True)[0] for masks in masks_all] 
        dlmbda = torch.autograd.grad(loss,lmbda,retain_graph=True,only_inputs=True)[0] 

        ## Apply layer norm
        ll_pxl_stable_all = [self.layer_norms[0](ll_pxl).detach() for ll_pxl in ll_pxl_all]
        dmu_x_stable_all = [self.layer_norms[1](dmu_x).detach() for dmu_x in dmu_x_all]
        dmasks_stable_all = [self.layer_norms[2](dmasks).detach() for dmasks in dmasks_all]
        dlmbda_stable = self.layer_norms[norm](dlmbda).detach()
        
        ## Generate coordinate channels
        H,W = (64,96)
        x_range = torch.linspace(-1.,1.,H)
        y_range = torch.linspace(-1.,1.,W)
        x_grid, y_grid = torch.meshgrid([x_range,y_range])
        x_grid =  x_grid.view((1, 1) + x_grid.shape).cuda()
        y_grid = y_grid.view((1, 1) + y_grid.shape).cuda()
        x_mesh = x_grid.expand(N,K,-1,-1,-1).contiguous()
        y_mesh = y_grid.expand(N,K,-1,-1,-1).contiguous()
        #### cat xxx_all
        _x_all = torch.cat(_x_all,dim=2)
        mu_x_all = torch.cat(mu_x_all, dim=2)
        masks_all = torch.cat(masks_all, dim=2)
        mask_logits_all = torch.cat(mask_logits_all, dim=2)
        dmu_x_stable_all = torch.cat(dmu_x_stable_all,dim=2)
        dmasks_stable_all = torch.cat(dmasks_stable_all, dim=2)
        p_masks_all = torch.cat(p_masks_all, dim=2)
        ll_pxl_stable_all = torch.cat(ll_pxl_stable_all,dim=2)


        ## Concatenate into vec and mat inputs
        img_args = (_x_all, mu_x_all,masks_all,mask_logits_all,dmu_x_stable_all,dmasks_stable_all,
                p_masks_all,ll_pxl_stable_all,x_mesh,y_mesh)
        vec_args = (lmbda, dlmbda_stable)
        

        img_inp = torch.cat(img_args,dim=2)
        vec_inp = torch.cat(vec_args,dim=1)

        ## Reshape
        img_inp = img_inp.view((N*K,)+img_inp.shape[2:])

        return {'img':img_inp, 'vec':vec_inp}

    """
    Computes the KL-divergence between an isotropic Gaussian distribution over latents
    parameterized by mu_z and logvar_z and the standard normal
    """
    def _get_div(self,mu_z,logvar_z,N,K):
            kl = ( -0.5*((1.+logvar_z-logvar_z.exp()-mu_z.pow(2)).sum(dim=1)) ).view((N,K))
            return (kl.sum(dim=1)).mean()

    """
    Implements the reparameterization trick
    Samples from standard normal and then scales and shifts by var and mu
    """
    def _sample(self,mu,logvar):
            std = torch.exp(0.5*logvar)
            return mu + torch.randn_like(std)*std


    """
    Loads weights for the IODINE model
    """
    def load(self,load_path,map_location='cpu'):
        model_dict = self.state_dict()
        state_dict = torch.load(load_path,map_location='cpu')
        new_state_dict = {key : state_dict[key] for key in state_dict if 'grid' not in key and 'z_dim' not in key}
        model_dict.update(new_state_dict)
        self.load_state_dict(model_dict)
        #print('load ckpt successfully')

    """
    Checks if any of the model's weights are NaN
    """
    def has_nan(self):
            for name,param in self.named_parameters():
                    if torch.isnan(param).any().item():
                            print(param)
                            assert False, '{} has nan'.format(name)

    """
    Checks if any of the model's weight's gradients are NaNs
    """
    def grad_has_nan(self):
            for name,param in self.named_parameters():
                    if torch.isnan(param.grad).any().item():
                            print(param)
                            print('---------')
                            print(param.grad)
                            assert False, '{}.grad has nan'.format(name)

class Interactor(nn.Module):
    def __init__(self, d_outer=12, d_inner=2): #hard code
        super(Interactor,self).__init__()
        self.d_outer = d_outer
        self.d_inner = d_inner
        self.linear_inter = nn.Linear(2*(d_outer + d_inner), 256)
        self.linear_inter_i = nn.Linear(2*(d_inner + 1 + d_outer  ), 256)
        self.linear_inter_a = nn.Linear(2*(d_outer + d_inner), 256)
        self.linear_inter_2 = nn.Linear(256, 256)
        self.linear_att = nn.Linear(256,100)
        self.linear_att_2 = nn.Linear(100,1)
        self.linear_S = nn.Sequential(nn.Linear(256, 256),nn.Linear(256, 1))
        self.linear_inter_C = nn.Linear(2*(d_outer + 1+d_inner), 256)
        self.linear_inter_i_C = nn.Linear(2*(d_inner + 1 + d_outer  ), 256)
        self.linear_inter_a_C = nn.Linear(2*(d_outer +1+ d_inner), 256)
        self.linear_inter_2_C = nn.Linear(256, 256)
        self.linear_att_C = nn.Linear(256,100)
        self.linear_att_2_C = nn.Linear(100,1)
        self.linear_S_C = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        self.norm_M = nn.BatchNorm1d(1)
        self.norm_C = nn.BatchNorm1d(1)
        self.norm = nn.BatchNorm1d(1)
        self.dynamic_update=nn.Linear(256, 2, bias=False)
        self.dynamic_update_C=nn.Linear(256, 2, bias=False)

    def forward(self,z_dyn,z_contex,z_mc,f_charge=1):
        z_charge = z_mc[:,1]
        z_mass = z_mc[:,0]
        contex_dynamic = torch.cat((z_dyn,z_contex),dim=-1).reshape(-1,8,self.d_inner+self.d_outer)
        x1 = contex_dynamic.unsqueeze(1).repeat(1, 8, 1, 1)
        x2 = contex_dynamic.unsqueeze(2).repeat(1, 1, 8, 1)
        m2 = z_mass.reshape(-1,8,1).unsqueeze(2).repeat(1, 1, 8, 1)
        m1 = z_mass.reshape(-1,8,1).unsqueeze(1).repeat(1, 8, 1, 1)
        v2 = z_dyn.reshape(-1,8,2).unsqueeze(2).repeat(1, 1, 8, 1)
        v1 = z_dyn.reshape(-1,8,2).unsqueeze(1).repeat(1, 8, 1, 1)

        ct2 = z_contex.reshape(-1,8,self.d_outer).unsqueeze(2).repeat(1, 1, 8, 1)
        ct1 = z_contex.reshape(-1,8,self.d_outer).unsqueeze(1).repeat(1, 8, 1, 1)
        ct_12 = torch.cat([ct1, ct2], -1)
        c2 = z_charge.reshape(-1,8,1).unsqueeze(2).repeat(1, 1, 8, 1)
        c1 = z_charge.reshape(-1,8,1).unsqueeze(1).repeat(1, 8, 1, 1)
        x_12 = torch.cat([x1, x2], -1)
        fc_12 = torch.cat([ct1,ct2,c1,c2,v1,v2], -1)
        f_12 = torch.cat([m1,x1,m2,x2], -1)
        E_emb = torch.relu(self.linear_inter(x_12))
        I_emb = torch.relu((self.linear_inter_i(f_12)))
        A_emb = torch.relu(self.linear_inter_a(x_12))
        E_emb_C = torch.relu(self.linear_inter_C(fc_12))
        I_emb_C = torch.relu(self.linear_inter_i_C(fc_12))
        A_emb_C = torch.relu(self.linear_inter_a_C(fc_12))
        presents=torch.sigmoid(self.linear_att_2(torch.tanh(self.linear_att(A_emb))))
        presents_C=torch.sigmoid(self.linear_att_2_C(torch.tanh(self.linear_att_C(A_emb_C))))
        mask = torch.ones_like(presents)
        for i in range(8):
            mask[:,i,i,:] = 0
        normed_scales = self.norm_M(self.linear_S(I_emb).permute(0, 3, 1, 2).contiguous().view(-1, 1, 64)).view(-1,1,8,8).permute(0, 2, 3, 1).contiguous()
        scale = (nn.Softplus()(normed_scales)*(presents*mask)).sum(1).flatten(0,1)
        normed_scales_C = self.norm_C(self.linear_S_C(I_emb_C).permute(0, 3, 1, 2).contiguous().view(-1, 1, 64)).view(-1,1,8,8).permute(0, 2, 3, 1).contiguous() 
        scale_C = (nn.Softplus()(normed_scales_C)*(mask)).sum(1).flatten(0,1)
        E = (self.linear_inter_2(E_emb)*(presents*mask)).sum(1)
        E_C = (self.linear_inter_2_C(E_emb_C)*(mask)).sum(1)
        new_dynamic_mass = scale*torch.nn.functional.normalize(self.dynamic_update(E).reshape(z_dyn.shape))
        new_dynamic_charge = scale_C*torch.nn.functional.normalize(self.dynamic_update_C(E_C).reshape(z_dyn.shape))
        new_dynamic = new_dynamic_mass + new_dynamic_charge
        
        return new_dynamic, new_dynamic_mass,new_dynamic_charge
