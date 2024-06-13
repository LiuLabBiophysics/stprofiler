import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import os.path as path
from os.path import split, join
from skimage.transform import rescale
from skimage.morphology import remove_small_objects
from skimage.measure import label
from tqdm import tqdm
import scipy.ndimage as ndi
from skimage.segmentation import expand_labels
import cv2

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def create_mask(anns, min_size, max_size, dapi_img, min_int, mask_separation):

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    mask = np.zeros(anns[0]['segmentation'].shape, dtype=np.uint32)

    good_mask_ind = np.full(len(anns), True)
    
    for i in range(len(sorted_anns)):
        if(not good_mask_ind[i]):
            continue
        for j in range(len(sorted_anns)):
            if(i == j):
                continue
            if(not good_mask_ind[j]):
                continue
            if(sorted_anns[j]['area'] < min_size):
                good_mask_ind[j] = False
                continue
            if(sorted_anns[j]['area'] > sorted_anns[i]['area']):
                continue
            if((np.logical_and(sorted_anns[i]['segmentation'], sorted_anns[j]['segmentation']).sum() / sorted_anns[j]['area']) > 0.9):
                if(mask_separation != 0):
                    new_mask = np.logical_and(sorted_anns[i]['segmentation'], 
                                            np.logical_not(expand_labels(sorted_anns[j]['segmentation'], mask_separation)))
                else:
                    new_mask = np.logical_and(sorted_anns[i]['segmentation'], 
                                            np.logical_not(sorted_anns[j]['segmentation']))
                
                new_mask = remove_small_objects(new_mask, min_size)
                sorted_anns[i]['segmentation'] = new_mask
                sorted_anns[i]['area'] = sorted_anns[i]['segmentation'].sum()

        if(sorted_anns[i]['area'] < min_size or sorted_anns[i]['area'] > max_size):
            good_mask_ind[i] =False
            

    for i in range(len(good_mask_ind)):
        if(good_mask_ind[i]):
            m = sorted_anns[i]['segmentation']
            cell_mask = dapi_img[m]
            cell_mean_int = cell_mask.mean()

            if(cell_mean_int <= min_int):
                continue
            
            mask = np.where(m, i+1, mask)
        
    return mask

def cell_segmentation(dapi_img, min_size, max_size, min_int,
                      cp_dir, points_per_side=128, points_per_batch=128,
                      scale_factor = None,
                      process_filter = None,
                      mask_separation = 0,
                      return_processed_img = False):
    
    orig_img_size_x = dapi_img.shape[0]
    orig_img_size_y = dapi_img.shape[1]
    
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint = cp_dir)
    sam.to(device = "cuda")

    if(scale_factor is not None):

        dapi_img = rescale(dapi_img, scale=1/scale_factor, 
                           order=1, anti_aliasing=True, preserve_range=True)

        min_size = int(min_size / scale_factor)
        max_size = int(max_size / scale_factor)

    dapi_img_uint8 = (dapi_img/256).astype(np.uint8)
    
    if(process_filter is not None):
        dapi_img_uint8 = cv2.filter2D(dapi_img_uint8, -1, process_filter).astype(np.uint8) 

    dapi_img_full = np.moveaxis(np.tile(dapi_img_uint8,[3,1,1]), 0, -1)
    
    mask_generator = SamAutomaticMaskGenerator(sam, 
                                               points_per_side=points_per_side, 
                                               points_per_batch=points_per_batch) 
    masks_dapi = mask_generator.generate(dapi_img_full)
    
    mask_img = create_mask(masks_dapi, min_size, max_size, dapi_img, min_int, mask_separation)

    if(scale_factor is not None):
        mask_img = rescale(mask_img, scale_factor, order=0)[:orig_img_size_x, :orig_img_size_y]

    if(return_processed_img):
        if(process_filter is not None):
            img_processed = rescale(dapi_img_uint8, scale_factor, order=0)[:orig_img_size_x, :orig_img_size_y]
        return mask_img.astype(np.uint32), img_processed
    else:
        return mask_img.astype(np.uint32) 


def cell_segmentation_tiles(dapi_img, 
                            min_size, max_size, 
                            tile_size, tile_overlap,
                            min_intensity,
                            cp_dir, points_per_side=128, points_per_batch=128,
                            scale_factor = None, 
                            process_filter = None,
                            return_processed_img = False,
                            img_offset_x_min = 0,
                            img_offset_x_max = 0,
                            img_offset_y_min = 0,
                            img_offset_y_max = 0):

    orig_img_size_x = dapi_img.shape[0]
    orig_img_size_y = dapi_img.shape[1]

    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint = cp_dir)
    sam.to(device = "cuda")

    if(scale_factor is not None):

        dapi_img = rescale(dapi_img, scale=1/scale_factor, 
                           order=1, anti_aliasing=True, preserve_range=True)

        min_size = int(min_size / scale_factor)
        max_size = int(max_size / scale_factor)

        tile_size = int(tile_size / scale_factor)
        tile_overlap = int(tile_overlap / scale_factor)

        img_offset_x_min = int(img_offset_x_min / scale_factor)
        img_offset_x_max = int(img_offset_x_max / scale_factor)
        img_offset_y_min = int(img_offset_y_min / scale_factor)
        img_offset_y_max = int(img_offset_y_max / scale_factor)
        
    max_intensity = np.max(dapi_img)
    max_uint8 = 255
    dapi_img_uint8 = ((dapi_img/max_intensity)*max_uint8).astype(np.uint8)
    mask_img = np.zeros(dapi_img.shape)
    
    if(process_filter is not None):
        dapi_img_uint8 = cv2.filter2D(dapi_img_uint8, -1, process_filter).astype(np.uint8) 

    for x in tqdm(range(0 + img_offset_x_min, dapi_img_uint8.shape[0] - img_offset_x_max, tile_size)):
        for y in range(0 + img_offset_y_min, dapi_img_uint8.shape[1] - img_offset_y_max, tile_size):

            x_max = np.min([dapi_img_uint8.shape[0], x+tile_size])
            y_max = np.min([dapi_img_uint8.shape[1], y+tile_size])

            x_tile_min = np.max([0, x-tile_overlap])
            y_tile_min = np.max([0, y-tile_overlap])

            x_tile_max = np.min([dapi_img_uint8.shape[0], x_max+tile_overlap])
            y_tile_max = np.min([dapi_img_uint8.shape[1], y_max+tile_overlap])

            if(np.mean(dapi_img_uint8[x_tile_min:x_tile_max, y_tile_min:y_tile_max]) <= min_intensity):
                continue

            dapi_img_full = np.moveaxis(np.tile(dapi_img_uint8[x_tile_min:x_tile_max, y_tile_min:y_tile_max],[3,1,1]), 0, -1)
                
            mask_generator = SamAutomaticMaskGenerator(sam, 
                                                       points_per_side=points_per_side, 
                                                       points_per_batch=points_per_batch) 
            masks_dapi = mask_generator.generate(dapi_img_full)

            mask_tile = create_mask(masks_dapi, min_size, max_size)

            mask_tile_resize = mask_tile[(x-x_tile_min):(-1*(x_tile_max-x_max)),
                                         (y-y_tile_min):(-1*(y_tile_max-y_max))]
        
            mask_tile_processed = remove_small_objects(mask_tile_resize, min_size)

            mask_img[x:x_max, y:y_max] =  mask_tile_processed

    #----------------------------------------------------------------------------------------

    mask_size = mask_img.shape
    nuc_mask_post = np.copy(mask_img)
    for i in range(100,mask_size[0], tile_size):
        for j in range(mask_size[1]):

            if(nuc_mask_post[i-1][j] != nuc_mask_post[i][j] and
               nuc_mask_post[i-1][j] != 0 and
               nuc_mask_post[i][j] != 0): 
                
                temp_mask = np.copy(nuc_mask_post[i-20:i+20,j-20:j+20])
                temp_mask_proc = np.where(temp_mask == nuc_mask_post[i][j], nuc_mask_post[i-1][j], temp_mask)
                nuc_mask_post[i-20:i+20,j-20:j+20] = temp_mask_proc

    for j in range(100,mask_size[1], tile_size):
        for i in range(mask_size[0]):

            if(nuc_mask_post[i][j-1] != nuc_mask_post[i][j] and
               nuc_mask_post[i][j-1] != 0 and 
               nuc_mask_post[i][j] != 0): 
                
                temp_mask = np.copy(nuc_mask_post[i-20:i+20,j-20:j+20])
                temp_mask_proc = np.where(temp_mask == nuc_mask_post[i][j], nuc_mask_post[i][j-1], temp_mask)
                nuc_mask_post[i-20:i+20,j-20:j+20] = temp_mask_proc

    if(scale_factor is not None):
        nuc_mask_post = rescale(nuc_mask_post, scale_factor, order=0)[:orig_img_size_x, :orig_img_size_y]

    if(return_processed_img):
        if(process_filter is not None):
            img_processed = rescale(dapi_img_uint8, scale_factor, order=0)[:orig_img_size_x, :orig_img_size_y]
        return nuc_mask_post.astype(np.uint32), img_processed
    else:
        return nuc_mask_post.astype(np.uint32)