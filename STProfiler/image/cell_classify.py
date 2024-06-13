import numpy as np
import pandas as pd
from ..feature.feature_extraction import savitzky_golay
from scipy.signal import argrelextrema

def cell_marker_classify(cell_label, marker_img, threshold_method='histogram'):
    
    cell_labels = np.unique(cell_labels)

    cell_df = pd.DataFrame()

    for cell in cell_labels:
        if cell==0:
            continue

        marker_img_cell = marker_img[cell_label==cell]

        marker_mean_int = marker_img_cell.mean()

        cell_df = pd.concat([cell_df, pd.DataFrame({'cell':cell, 'mean_marker_intensity':marker_mean_int})])

    
    if(threshold_method == 'histogram'): # Assuming two distinct population
        hist = np.histogram(cell_df['mean_marker_intensity'])
        threshold = hist[1][argrelextrema(savitzky_golay(hist[0], 13, 2), np.less)[0][0]]
    elif(threshold_method == 'median'): 
        threshold = cell_df['mean_marker_itensity'].median()
    elif(threshold_method=='mean'):
        threshold = cell_df['mean_marker_intensity'].mean()
    else:
        raise ValueError('Threshold method must be "histogram", "median" or "mean"')

    cell_df['over_thres'] = cell_df['mean_marker_intensity'] >= threshold

    return cell_df

    