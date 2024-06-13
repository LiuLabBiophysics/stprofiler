from skimage.feature import blob_log
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def rna_detection(tif_rna, 
                  min_sig=1, 
                  max_sig=5, 
                  num_sig=5, 
                  thres=0.001,
                  tile_size=500,
                  border_exclude=10):

    blobs_df = pd.DataFrame()

    for x in range(border_exclude,tif_rna.shape[0]-border_exclude, tile_size):
        for y in range(border_exclude,tif_rna.shape[0]-border_exclude, tile_size):

            x_max = np.min([tif_rna.shape[0]-border_exclude, x+tile_size+border_exclude])
            y_max = np.min([tif_rna.shape[1]-border_exclude, y+tile_size+border_exclude])

            image_tile = tif_rna[x-border_exclude:x_max][y-border_exclude:y_max]

            blobs_array  = blob_log(image_tile,
                                    min_sigma=min_sig,
                                    max_sigma=max_sig,
                                    num_sigma=num_sig,
                                    threshold=thres,
                                    overlap=0.01,
                                    exclude_border=border_exclude)

            blobs_df = pd.concat([blobs_df, pd.DataFrame({'x':blobs_array[:,0] + x - border_exclude, 'y':blobs_array[:,1] + y - border_exclude})])

    return blobs_df

def plot_rna(tif_rna, blobs_df):
    
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2, sharex=ax1, sharey=ax1)

    ax1.imshow(tif_rna)

    ax2.imshow(tif_rna)
    ax2.scatter(blobs_df['y'], blobs_df['x'], marker='x', color='r')

    plt.tight_layout()
    plt.show()

    return