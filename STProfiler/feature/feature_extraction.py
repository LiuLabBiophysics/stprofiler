import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from numpy.linalg import norm

#----------------------------------------------------------------------------------------------------

#   Feature dataframe integration

#----------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------

#   Single foci feature

#----------------------------------------------------------------------------------------------------

def single_foci_feature(rna_df, nuc_label, cell_label, points_dist, max_search_range=300):
    
    rna_df['label'] = cell_label[(rna_df['x'].to_numpy().astype(int)),
                                 (rna_df['y'].to_numpy().astype(int))]
    
    img_size = cell_label.shape[0]

    pixel = pd.DataFrame()
    pixel_nuc = pd.DataFrame()
    x_list = []
    y_list = []
    cell_label_list = []
    x_nuc_list = []
    y_nuc_list = []
    cell_label_nuc_list = []

    for a in range(cell_label.shape[0]):
        for b in range(cell_label.shape[1]):
            x_list.append(a)
            y_list.append(b)
            cell_label_list.append(cell_label[a][b])

            if(nuc_label[a][b] != 0):
                x_nuc_list.append(a)
                y_nuc_list.append(b)
                cell_label_nuc_list.append(nuc_label[a][b])    

    pixel['x'] = x_list
    pixel['y'] = y_list
    pixel['label'] = cell_label_list

    pixel_nuc['x'] = x_nuc_list
    pixel_nuc['y'] = y_nuc_list
    pixel_nuc['label'] = cell_label_nuc_list


    # centroid = pixel.groupby(['label']).mean()
    # centroid_nuc = pixel_nuc.groupby(['label']).mean()

    features = pd.DataFrame()
    cell_label_list = []
    xs = []
    ys = []
    f1 = []                # f1: closest distance to cell outline
    f2 = []                # f2: distance to cell centroid
    f3 = []                # f3: Distance to nuclear centroid
    f4 = []                # f4: Radius to include 5%  off all remaining spots in the cell
    f5 = []                # f5: Radius to include 10% off all remaining spots in the cell
    f6 = []                # f6: Radius to include 15% off all remaining spots in the cell
    f7 = []                # f7: Radius to include 25% off all remaining spots in the cell
    f8 = []                # f8: Radius to include 50% off all remaining spots in the cell
    f9 = []                # f9: Radius to include 75% off all remaining spots in the cell
    f10 = []               # f10: Fraction of remaining spots at 20  pixels
    f11 = []               # f11: Fraction of remaining spots at 40  pixels
    f12 = []               # f12: Fraction of remaining spots at 80  pixels
    f13 = []               # f13: Fraction of remaining spots at 120 pixels
    f14 = []               # f14: Mean distance to all other spots
    f15 = []               # f15: Standard deviation of distances to all other spots
    f16 = []               # f16: Variance of distances to all other spots

    # for n, row in tqdm(rna_df.iterrows(), total=rna_df.shape[0]):
    for n, row in rna_df.iterrows():

        x = int(row['x'])
        y = int(row['y'])
        dot_label = int(cell_label[x][y])

        rna_df_cell = rna_df[rna_df['label'] == dot_label]
        
        if(rna_df_cell.shape[0] == 1):
            continue

        xs.append(x)
        ys.append(y)
        cell_label_list.append(dot_label)

        #calculate distance to centroids
        centroid_x = pixel[pixel['label'] == dot_label]['x'].mean()
        centroid_y = pixel[pixel['label'] == dot_label]['y'].mean()

        centroid_nuc_x = pixel_nuc[pixel_nuc['label'] == dot_label]['x'].mean()
        centroid_nuc_y = pixel_nuc[pixel_nuc['label'] == dot_label]['y'].mean()

        distance_cent = np.sqrt((x-centroid_x)**2 + (y-centroid_y)**2)
        distance_cent_nuc = np.sqrt((x-centroid_nuc_x)**2 + (y-centroid_nuc_y)**2)

        f2.append(distance_cent)
        f3.append(distance_cent_nuc)

        

        #radial search
        C1_set = set()
        dist = np.zeros(rna_df_cell.shape[0]-1)
        jj=0
        for j, row2 in rna_df_cell.iterrows():
            C1_set.add(int(row2['x'])*10000 + int(row2['y']))
            if (j != n):
                dist[jj] = np.sqrt((row['x']-row2['x'])**2 + (row['y']-row2['y'])**2)
                jj = jj+1

        f14.append(dist.mean())
        f15.append(np.std(dist))
        f16.append(np.var(dist))

        count = 0
        reached_border = False
        reached_5 = False
        reached_10 = False
        reached_15 = False
        reached_25 = False
        reached_50 = False
        reached_75 = False

        rna_num = rna_df_cell.shape[0]
        for k in range(max_search_range-1):
            for l in range(len(points_dist[k])):
                point = points_dist[k][l]
                if(((x + point[0])*10000 + (y + point[1])) in C1_set):
                    count = count + 1
                if(((x + point[0])*10000 + (y - point[1])) in C1_set):
                    count = count + 1
                if(((x - point[0])*10000 + (y + point[1])) in C1_set):
                    count = count + 1
                if(((x - point[0])*10000 + (y - point[1])) in C1_set):
                    count = count + 1

                if(not reached_border):
                    if((x+point[0]<img_size) and (y+point[1]<img_size) and (cell_label[x + point[0]][y + point[1]] != dot_label)):
                        f1.append(k+1)
                        reached_border = True
                    elif((x+point[0]<img_size) and (y-point[1]>=0) and (cell_label[x + point[0]][y - point[1]] != dot_label)):
                        f1.append(k+1)
                        reached_border = True
                    elif((x-point[0]>=0) and (y+point[1]<img_size) and (cell_label[x - point[0]][y + point[1]] != dot_label)):
                        f1.append(k+1)
                        reached_border = True
                    elif((x+point[0]>=0) and (y-point[1]>=0) and (cell_label[x - point[0]][y - point[1]] != dot_label)):
                        f1.append(k+1)
                        reached_border = True

            if(k+1 == 20):
                f10.append(count/rna_num)
            if(k+1 == 40):
                f11.append(count/rna_num)
            if(k+1 == 80):
                f12.append(count/rna_num)
            if(k+1 == 120):
                f13.append(count/rna_num)

            if((not reached_5) and (0.05*rna_num <= count)):
                reached_5 = True
                f4.append(k+1)

            if((not reached_10) and (0.10*rna_num <= count)):
                reached_10 = True
                f5.append(k+1)

            if((not reached_15) and (0.15*rna_num <= count)):
                reached_15 = True
                f6.append(k+1)

            if((not reached_25) and (0.25*rna_num <= count)):
                reached_25 = True
                f7.append(k+1)

            if((not reached_50) and (0.50*rna_num <= count)):
                reached_50 = True
                f8.append(k+1)

            if((not reached_75) and (0.75*rna_num <= count)):
                reached_75 = True
                f9.append(k+1)
        if(not reached_border):
            f1.append(max_search_range)
        if(not reached_5):
            f4.append(max_search_range)
        if(not reached_10):
            f5.append(max_search_range)
        if(not reached_15):
            f6.append(max_search_range)
        if(not reached_25):
            f7.append(max_search_range)
        if(not reached_50):
            f8.append(max_search_range)
        if(not reached_75):
            f9.append(max_search_range)

    features['cell'] = cell_label_list
    features['x'] = xs
    features['y'] = ys
    features['feature 1'] = f1
    features['feature 2'] = f2
    features['feature 3'] = f3
    features['feature 4'] = f4
    features['feature 5'] = f5
    features['feature 6'] = f6
    features['feature 7'] = f7
    features['feature 8'] = f8
    features['feature 9'] = f9
    features['feature 10'] = f10
    features['feature 11'] = f11
    features['feature 12'] = f12
    features['feature 13'] = f13
    features['feature 14'] = f14
    features['feature 15'] = f15
    features['feature 16'] = f16
    
    return features

def create_points_dist(max_search_range=300):
    points_dist = []

    for i in range(1,max_search_range):
        points = []
        for j in range(i+1):
            for k in range(i+1):
                if(np.ceil(np.sqrt((j**2 + k**2))) == (i)):
                    points.append([j,k])
        points_dist.append(points)

    return points_dist

#----------------------------------------------------------------------------------------------------

#   Ripley's H feature

#----------------------------------------------------------------------------------------------------

def ripleyH_feature(rna_df, cell_label, plane, search_range_rH=100, use_edge_correction=False):

    img_size = cell_label.shape[0]

    labels = np.unique(cell_label).astype(int)

    x = rna_df['x']
    y = rna_df['y']
    rna_df['label'] = cell_label[np.array(x).astype(int), np.array(y).astype(int)]

    H_tot = pd.DataFrame()
    for l in labels:

        rna_df_cell = rna_df[rna_df['label'] == l].reset_index()
        
        if(l == 0):
            continue

        # if(rna_df_cell.shape[0] > 500):
        #     H = np.zeros(search_range_rH)
        #     H_series = pd.DataFrame(H).T.assign(cell=l)
        #     H_tot = pd.concat([H_tot, H_series])
        #     continue

        if((rna_df_cell.shape[0] > 500) or (rna_df_cell.shape[0] == 0)):
            continue

        label_cell = np.where(cell_label == l, 255, 0)
        foci_map = np.zeros((rna_df_cell.shape[0], search_range_rH)) 
        H = []
        for i,row1 in rna_df_cell.iterrows():
            x1 = int(row1['x'])
            y1 = int(row1['y'])

            for j,row2 in rna_df_cell.iterrows():
                x2 = int(row2['x'])
                y2 = int(row2['y'])

                dist = np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2)).astype(np.int32)

                if(dist!=0 and dist<search_range_rH):
                    if(use_edge_correction):
                        w = edge_correction([x1,y1],dist,label_cell, img_size, plane, search_range_rH)
                        foci_map[i, dist] += 1/w
                    else:
                        foci_map[i, dist] += 1
        
        area = ndi.sum_labels(np.ones(cell_label.shape), cell_label, index=l)
        lambda1 = rna_df_cell.shape[0] / area  
        for i in range(search_range_rH):
            n = 0
            for j in range(rna_df_cell.shape[0]):
                n += foci_map[j,1:i+1].sum()

            H.append(np.sqrt(n/(lambda1**2 * area)/np.pi)-i)

        H_series = pd.DataFrame(H).T.assign(cell=l)
        H_tot = pd.concat([H_tot, H_series])

    return H_tot

def ripleyH_create_plane(search_range_rH=100):

    plane = np.zeros((search_range_rH*2,search_range_rH*2,2))
    for i in range(search_range_rH*2):
        for j in range(search_range_rH*2):
            plane[i,j,0] = i - search_range_rH
            plane[i,j,1] = j - search_range_rH  

    return plane


def edge_correction(coor,r,mask,img_size,plane,search_range_rH):

        img_plane = np.zeros((img_size + 2*search_range_rH,img_size + 2*search_range_rH))
        extend_mask = np.zeros((img_size + 2*search_range_rH,img_size + 2*search_range_rH))
        extend_mask[search_range_rH:-search_range_rH, search_range_rH:-search_range_rH] = mask

        circ = np.where(np.linalg.norm(plane, axis=(2))<r, 1, 0)

        img_plane[coor[0]: coor[0]+2*search_range_rH,
                  coor[1]: coor[1]+2*search_range_rH] = circ
        area_in = (np.where(extend_mask==255, img_plane, 0)).sum()
        area = circ.sum()

        return area_in/area

def rH_extrema_feature(H_tot,
                       use_max_loc = True,
                       use_min_loc = True,
                       use_diff_max_loc = True,
                       use_diff_min_loc = True,
                       smooth_window_size=9,
                       smooth_order=4):
    
    rH_array = H_tot.drop(columns='cell').to_numpy()
    rH_array_smooth = np.zeros(rH_array.shape)

    for i in range(rH_array.shape[0]):
        rH_array_smooth[i] = savitzky_golay(rH_array[i], smooth_window_size, smooth_order)

    rH_extr_feature = pd.DataFrame()

    rH_extr_feature['cell'] = H_tot['cell'].tolist()

    if(use_max_loc):
        rH_extr_feature['rH max loc'] = np.argmax(rH_array_smooth[:,:-10], axis=1)

    if(use_min_loc):
        rH_extr_feature['rH min loc'] = np.argmin(rH_array_smooth[:,:-10], axis=1)

    if(use_diff_max_loc):
        rH_extr_feature['rH diff max loc'] = np.argmax(np.diff(rH_array_smooth)[:,:-10], axis=1)

    if(use_diff_min_loc):
        rH_extr_feature['rH diff min loc'] = np.argmin(np.diff(rH_array_smooth)[:,:-10], axis=1)

    return rH_extr_feature



def rH_feature_avg(H_tot, interval=5, max_range=100):

    rH_df = H_tot.drop(columns=['cell'])

    rH_num = rH_df.shape[0]

    if(max_range % interval != 0):
        cols = rH_df.columns
        rH_df.drop(columns=cols[-1*(max_range%interval)])

    rH_features = pd.DataFrame()

    for i in range(rH_num):
        rH_row = rH_df.iloc[i].to_numpy()

        rH_row_avg = rH_row.reshape(-1,interval).mean(axis=1)
        
        rH_features = pd.concat([rH_features, pd.DataFrame(rH_row_avg).T], axis=0, ignore_index=True)

    new_col = []

    for num in range(0, max_range, interval):
        new_col.append('rH ' + str(num + interval))

    rH_features.columns = new_col

    rH_features['cell'] = H_tot['cell'].tolist()

    return rH_features

#----------------------------------------------------------------------------------------------------

#   Boundary clustering feature

#----------------------------------------------------------------------------------------------------

def bound_feature(rna_df, nuc_label, cell_label, search_range_bound=100):

    nuc_mask = nuc_label!=0

    x = rna_df['x']
    y = rna_df['y']
    rna_df['at nucleus'] = nuc_label[np.array(x).astype(int), np.array(y).astype(int)] != 0

    rna_df_nuc = rna_df[rna_df['at nucleus']]
    rna_df_cyto = rna_df[rna_df['at nucleus']==False]

    x_nuc  = np.array(rna_df_nuc['x']).astype(int)
    y_nuc  = np.array(rna_df_nuc['y']).astype(int)
    x_cyto = np.array(rna_df_cyto['x']).astype(int)
    y_cyto = np.array(rna_df_cyto['y']).astype(int)

    rna_df_nuc = rna_df_nuc.assign(label=cell_label[x_nuc, y_nuc])
    rna_df_cyto = rna_df_cyto.assign(label=cell_label[x_cyto, y_cyto])

    bound_nums_inner = pd.DataFrame()
    bound_nums_outer = pd.DataFrame()

    for l in np.unique(cell_label):
        label_cell = (cell_label==l)
        label_nuc = (label_cell & nuc_mask)
        dist_cell_inner = ndi.distance_transform_edt(label_nuc)
        dist_cell_outer = ndi.distance_transform_edt(1-label_nuc)
        det_cell_nuc = rna_df_nuc[rna_df_nuc['label']==l]
        det_cell_cyto = rna_df_cyto[rna_df_cyto['label']==l]
        dist_miR_inner = dist_cell_inner[np.array(det_cell_nuc['x']).astype('int'),
                                        np.array(det_cell_nuc['y']).astype('int')]
        dist_miR_outer = dist_cell_outer[np.array(det_cell_cyto['x']).astype('int'),
                                        np.array(det_cell_cyto['y']).astype('int')]
        
        curr_nums_inner = np.zeros(search_range_bound)
        curr_nums_outer = np.zeros(search_range_bound)
        for i in range(search_range_bound):
            curr_nums_inner[i] = len(np.where(dist_miR_inner < i+1)[0])
            curr_nums_outer[i] = len(np.where(dist_miR_outer < i+1)[0])
            
        bound_nums_inner = pd.concat([bound_nums_inner, pd.DataFrame(curr_nums_inner.reshape(1,-1)).assign(cell=l)], axis=0)
        bound_nums_outer = pd.concat([bound_nums_outer, pd.DataFrame(curr_nums_outer.reshape(1,-1)).assign(cell=l)], axis=0)
        
    return bound_nums_inner, bound_nums_outer

def bound_feature_avg(bound_tot, inner, interval=5, max_range=100):

    if(inner):
        feature_name = 'inner num '
    else:
        feature_name = 'outer num '

    bound_df = bound_tot.drop(columns=['cell'])

    bound_num = bound_df.shape[0]
    max_range = bound_df.shape[1]

    if(max_range % interval != 0):
        cols = bound_df.columns
        bound_df.drop(columns=cols[-1*(max_range%interval)])

    bound_features = pd.DataFrame()

    for i in range(bound_num):
        bound_row = bound_df.iloc[i].to_numpy()

        bound_row_avg = bound_row.reshape(-1,interval).mean(axis=1)
        
        bound_features = pd.concat([bound_features, pd.DataFrame(bound_row_avg).T], axis=0, ignore_index=True)

    new_col = []

    for num in range(0, max_range, interval):
        new_col.append('bound ' + feature_name + str(num + interval))

    bound_features.columns = new_col

    bound_features['cell'] = bound_tot['cell'].tolist()

    return bound_features

#----------------------------------------------------------------------------------------------------

#   RDI feature

#----------------------------------------------------------------------------------------------------

def RDI_features(rna_df, cell_label):

    x = np.array(rna_df['x']).astype(int)
    y = np.array(rna_df['y']).astype(int)

    rna_df = rna_df.assign(label=cell_label[x, y])

    RDI_df = pd.DataFrame()

    for l in np.unique(cell_label):
        label_cell = cell_label == l
        rna_df_cell = rna_df[rna_df['label']==l]

        if(rna_df_cell.empty):
            continue

        cell_mean = np.mean(np.where(label_cell), axis=1)
        rna_cell_mean = np.mean(np.array(rna_df_cell[['x', 'y']]), axis=0)

        cell_pixel = np.array(np.where(label_cell)).T
        rna_cell_pixel = np.array(rna_df_cell[['x', 'y']])

        Rg = np.sqrt(np.sum(norm(cell_pixel - cell_mean, axis=1)) / cell_pixel.shape[0])

        PI = norm(rna_cell_mean - cell_mean)/Rg

        DI = (np.sum(norm(rna_cell_pixel - rna_cell_mean, axis=1)**2) / rna_cell_pixel.shape[0]) / \
            (np.sum(norm(cell_pixel - rna_cell_mean, axis=1)**2) / cell_pixel.shape[0])
        
        PDI = (np.sum(norm(rna_cell_pixel - cell_mean, axis=1)**2) / rna_cell_pixel.shape[0]) / \
            (np.sum(norm(cell_pixel - cell_mean, axis=1)**2) / cell_pixel.shape[0])
        
        RDI_df = pd.concat([RDI_df, pd.DataFrame({'cell':l, 'PI':PI, 'DI':DI, 'PDI':PDI}, index=[0])])

    return RDI_df


#----------------------------------------------------------------------------------------------------

#   Misc. functions

#----------------------------------------------------------------------------------------------------

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
#     values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')