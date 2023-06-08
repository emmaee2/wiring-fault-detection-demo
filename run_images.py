import cv2
import matplotlib.pyplot as plt
import numpy as np
from model.model import MapModel
# from train import varchecks
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from os import listdir
from os.path import isfile, join
import copy
from skimage.morphology import square, binary_closing
from scipy import ndimage
import random


def single_inference(im_path, symbol_path, ckpt_dir, binary_classification=False, distance = 'euc'):
    M = 64
    N = 64

    im = cv2.imread(im_path).astype('float32')/255
    # mask = cv2.imread(mask_path)

    #tile image patch
    tiles = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]

    dataloader = torch.utils.data.DataLoader(tiles, batch_size=1, 
                                                shuffle=False, 
                                                num_workers=1)
    device = 'cuda:3'
    img_shape = (3, M, N)

    model = MapModel(img_shape, bn=True, share_encoder=False, binary_classification=binary_classification)
    model.load_state_dict(torch.load(ckpt_dir))
    model = model.to(device)
    model.eval()

    symbol = cv2.imread(symbol_path).astype('float32')/255
    symbol = torch.from_numpy(symbol).permute(2, 0, 1)
    symbol = symbol[None, :, :, :]

    tiles_out = []
    embeddings_out = []
    for batch_idx, map_patches in enumerate(dataloader):
        map_patches = map_patches.permute(0,3, 1, 2)

        symbol = symbol.to(device)
        map_patches = map_patches.to(device)

        symbols_out, _ = model(symbol, None)
        bsize, f = symbols_out.shape
        symbols_out = symbols_out.view(bsize,1 ,1, f)
        
        _, map_patches_out = model(None, map_patches)
        # map_patches_out: (batch, features, h, w)
        if distance == 'euc':
            euclidean_dist = ( symbols_out - map_patches_out.permute(0, 2, 3, 1)).pow(2).sum(-1).sqrt()
        elif distance == 'cos':
            a = symbols_out
            b = map_patches_out.permute(0, 2, 3, 1)
            cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
        
        if binary_classification == True:
            logits = model.classifier(euclidean_dist[0])
            preds = F.sigmoid(logits)
            tiles_out.append(preds)   
        else:
            tiles_out.append(euclidean_dist[0])
        embeddings_out.append(map_patches_out.permute(0, 2, 3, 1))
        
    # euclidean_dist.cpu().detach().numpy()[0]
    
    embeddings_assembled = np.zeros((np.shape(im)[0],np.shape(im)[1],64))
    tiles_assembled = np.zeros(np.shape(im)[:-1])
    tcnt = 0
    for x in range(0,im.shape[0],M):
        for y in range(0,im.shape[1],N):
            tiles_assembled[x:x+M,y:y+N] = tiles_out[tcnt].cpu().detach().numpy()
            embeddings_assembled[x:x+M,y:y+N,:] = embeddings_out[tcnt].cpu().detach().numpy()
            tcnt += 1
    return tiles_assembled, symbols_out, embeddings_assembled 

def encoder_output(im_path):
    M = 64
    N = 64

    im = cv2.imread(im_path).astype('float32')/255
    # mask = cv2.imread(mask_path)

    #tile image patch
    tiles = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]

    dataloader = torch.utils.data.DataLoader(tiles, batch_size=1, 
                                                shuffle=False, 
                                                num_workers=1)
    device = 'cuda:3'
    img_shape = (3, M, N)

    model = MapModel(img_shape, bn=True, share_encoder=False)
    model.load_state_dict(torch.load(ckptdir))
    model = model.to(device)
    model.eval()

    symbol = cv2.imread(symbol_path).astype('float32')/255
    symbol = torch.from_numpy(symbol).permute(2, 0, 1)
    symbol = symbol[None, :, :, :]

    tiles_out = []
    for batch_idx, map_patches in enumerate(dataloader):
        map_patches = map_patches.permute(0,3, 1, 2)

        symbol = symbol.to(device)
        map_patches = map_patches.to(device)

        symbols_out, _ = model(symbol, None)
        bsize, f = symbols_out.shape
        symbols_out = symbols_out.view(bsize,1 ,1, f)
        _, map_patches_out = model(None, map_patches)
        # map_patches_out: (batch, features, h, w)

        euclidean_dist = ( symbols_out - map_patches_out.permute(0, 2, 3, 1)).pow(2).sum(-1).sqrt()

        tiles_out.append(euclidean_dist[0])
    euclidean_dist.cpu().detach().numpy()[0]

    tiles_assembled = np.zeros(np.shape(im)[:-1])
    tcnt = 0
    for x in range(0,im.shape[0],M):
        for y in range(0,im.shape[1],N):
            tiles_assembled[x:x+M,y:y+N] = tiles_out[tcnt].cpu().detach().numpy()
            tcnt += 1
    return enc


def run_inference_multiple_symbols(im_path, symbol_paths, ckpt_dir, num_syms = 10):
    euc_dists = []
    symb_embeds = []
    patch_embeds = []
    random.shuffle(symbol_paths)
    for symbol_path in symbol_paths[:num_syms]:
        # print(os.path.basename(symbol_path))
        euc_dist, symb_embed, patch_embed = single_inference(im_path, symbol_path, ckpt_dir, binary_classification=False)
        euc_dists.append(euc_dist)
        symb_embeds.append(symb_embed)
        patch_embeds.append(patch_embed)
        
    euc_sum = np.zeros(np.shape(euc_dists[0]))
    for i in range(len(euc_dists)):
        euc_sum += euc_dists[i]
    return euc_sum


def thresh(image, thresh):
    im = copy.deepcopy(image)
    im[im >= thresh] = 1000
    im[im < thresh] = 1
    im[im == 1000] = 0
    return im

def clean_prediction(euc_sum, threshold = 0.1, square_size = 5, cc_cutoff = 500/2):
    #threshold -> binary closing -> remove small components
    pred_mask = np.zeros(np.shape(euc_sum), dtype = int)
    thresh_mask = thresh(euc_sum, threshold)
    pred_mask[thresh_mask > 0] = 255
    
    pred_mask_c = binary_closing(pred_mask, square(square_size))
    
    Zlabeled,Nlabels = ndimage.label(pred_mask_c)
    label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]
    for label,size in enumerate(label_size):
        if size < cc_cutoff:
            pred_mask_c[Zlabeled == label] = 0
    
    # pred_mask_c = binary_closing(pred_mask, square(square_size*5))
    return pred_mask_c
            

def mask_size(train_mask_dir):
    train_mask_paths = [os.path.join(train_mask_dir, f) for f in os.listdir(train_mask_dir) ]

    label_sizes = []
    for i in range(len(train_mask_paths)):
        mask = cv2.imread(train_mask_paths[i])[:,:,0]
        Zlabeled,Nlabels = ndimage.label(mask)
        label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]
        for label,size in enumerate(label_size): 
            label_sizes.append(size)        
    return np.min(label_sizes)

def score_one_image(pred, mask_path, min_mask_size = 500, overlap = 0.5):
    #count the number of labels mask
    mask = cv2.imread(mask_path)[:,:,0]
    Zlabeled,Nlabels = ndimage.label(mask)
    
    n_corr_preds = 0
    n_preds = 0
    #for each discrete object see how much overlap it has with prediction
    for label in range(1,Nlabels+1):
        total_pnts = (Zlabeled == label).sum()
        pnts_covered = pred[Zlabeled == label].sum()
        #if overlap > 0.3 ncorrpreds+= 1
        if pnts_covered > (min_mask_size * overlap):
            n_corr_preds += 1
    
    #count the number of labels predictions
    Zlabeled,Nlabels_pred = ndimage.label(pred)
    for label in range(1,Nlabels_pred+1):
        pnts_covered = pred[Zlabeled == label].sum()
        #if overlap > 0.3 ncorrpreds+= 1
        if pnts_covered > (min_mask_size * overlap):
            n_preds += 1
        else:
            pred[Zlabeled == label] = 0
    
    #return the number of predicted faults, number of correctly predicted faults, and number of total faults
    return Nlabels_pred, n_corr_preds, Nlabels, pred