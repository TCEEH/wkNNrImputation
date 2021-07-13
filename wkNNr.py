from multiprocessing import Pool
import numpy as np
import pandas as pd
import logging

from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.model_selection import KFold
import csv
import time


'''The main function available in the following code is 'impute_wkNNr_parallel'.
The function recieves the following arguments:
corr = the correlation matrix, as calculated from the unimputed dataset. 
missing_data = the unimputed dataset. 
batch_size = imputation is performed in batches, batch_size stands for the number of rows to impute in a batch.
k_range = k parameter of the kNN algorithm. A numpy array of size 1 or larger, in case several imputations by different k values are desired. 
chunk_size = the size of a work unit in a parallel computation. A low chunk_size requires more memory but will less likely to create a buttleneck. 
njobs = number of jobs to run in parallel. None (defualt) means use all computer cores.

The function returns:
imputed - a 3D matrix. axis 0 = the index of the k parameter in k_range, axis 1 = rows of imputed matrix and axis 2 = columns of imputed matrix. 

'''

def impute_wkNNr_parallel(corr, missing_data, batch_size, k_range, corr_thr=0, parallel=True, chunk_size=1, njobs=None):
    imputed = np.repeat(np.asarray(missing_data)[None], k_range.shape[0], axis=0)
    args = ((missing_data, corr, i, batch_size, k_range, corr_thr) for i in range(missing_data.shape[1]))
    if parallel: 
        with Pool(njobs) as p:
            for rinds, cind, total_values in p.imap_unordered(impute_column, args, chunk_size):
                for k_idx, _ in enumerate(k_range):
                    imputed[k_idx, rinds, cind] = total_values[:, k_idx]
    else: 
        for rinds, cind, total_values in map(impute_column, args):
                for k_idx, _ in enumerate(k_range):
                    imputed[k_idx, rinds, cind] = total_values[:, k_idx]
    return imputed

'''impute_wkNNr_parallel if run with an enlarged dataset which includes the lag and leads.
c_start = the first column of the desired columns to impute
c_end = the last column of the desired columns to impute
* the correlation matrix required as input is the correlation matrix calculated on the enlarged dataset. 
'''

def impute_wkNNr_parallel_with_add(corr, missing_data, batch_size, k_range, corr_thr, c_start, c_end, parallel=True, chunk_size=1, njobs=None):
    imputed = np.repeat(np.asarray(missing_data)[None], k_range.shape[0], axis=0)
    args = ((missing_data, corr, i, batch_size, k_range, corr_thr) for i in range(c_start, c_end))
    if parallel: 
        with Pool(njobs) as p:
            for rinds, cind, total_values in p.imap_unordered(impute_column, args, chunk_size):
                for k_idx, _ in enumerate(k_range):
                    imputed[k_idx, rinds, cind] = total_values[:, k_idx]
    else: 
        for rinds, cind, total_values in map(impute_column, args):
                for k_idx, _ in enumerate(k_range):
                    imputed[k_idx, rinds, cind] = total_values[:, k_idx]
    return imputed


def return_standardized(mat):
    mat / np.nanstd(mat, axis=0) 
    return mat


def impute_column(args):
    missing_data, corr, cind, batch_size, k_range, corr_thr = args
    start_time = time.time()
    logging.info(f'starting {cind}')
    rinds = np.isnan(missing_data[:, cind])

    # Use threshold of corr:
    correlated_columns = np.abs(corr[:, cind]) > corr_thr
    logging.info(f'{correlated_columns.sum()} correlated columns')
    cind = correlated_columns[:cind].sum()
    corr = corr[correlated_columns, :][:, correlated_columns]
    missing_data = missing_data[:, correlated_columns]

    # standardize data
    missing_data_norm = return_standardized(missing_data)
    missing_rows = missing_data_norm[rinds, :]
    present_rows = missing_data[~rinds, :]
    present_rows_norm = missing_data_norm[~rinds, :]
    present_rows_corr = present_rows_norm * corr[:, cind]
    present_rows_not_nan_mask = (~np.isnan(present_rows)).astype(int)

    total_values = np.zeros((missing_rows.shape[0], len(k_range)))
    for batch_start in range(0, missing_rows.shape[0], batch_size):
        batch = missing_rows[batch_start:batch_start + batch_size]

        weights = get_weights(batch, cind, corr, present_rows_corr, present_rows_not_nan_mask)

        get_values_for_weights(total_values, weights, cind, k_range, present_rows, batch_start, batch_size)

    logging.info(f'finished {cind} ({(time.time() - start_time)/60} minutes)')
    return rinds, cind, total_values


def get_weights(batch, cind, corr, present_rows_corr, present_rows_not_nan_mask):
    distances = nan_euclidean_distances(present_rows_corr, batch * corr[:, cind])
    counts = np.dot(present_rows_not_nan_mask, (~np.isnan(batch)).T.astype(int))
    weights = (counts**(2)) / distances #from nan_euclidean_distances we get sqrt(counts) (in addition)
    return weights


def get_values_for_weights(total_values, weights, cind, k_range, present_rows, batch_start, batch_size):
    max_k = max(k_range)
    thrs = np.partition(weights, -(max_k + 1), axis=0)[-(max_k + 1):, :]
    thrs[np.isnan(thrs)] = 0
    thrs = np.sort(thrs, axis=0)[::-1, :]
    weights[np.isnan(weights)] = 0
    for k_idx, k_neigh in enumerate(k_range):
        logging.info(f'k_neighbors {k_neigh}')
        min_thr = thrs[k_neigh, :]
        inds = weights > min_thr
        rows = inds.any(axis=1).squeeze()
        weights_copy = np.zeros((rows.sum(), weights.shape[1]))
        weights_copy[inds[rows,:]] = (weights[rows,:] - min_thr)[inds[rows,:]]
        weights_copy = weights_copy / weights_copy.sum(axis=0)

        total_values[batch_start:batch_start + batch_size, k_idx] = np.dot(present_rows[rows, cind], weights_copy)



        
        
        
        
        
        
'''Generate dummy data to test imputation functions'''

def get_column(length, frequency, offset, noise_level):
    noise = np.random.normal(size=length)
    return np.sin(np.linspace(offset, frequency * 2 * np.pi + offset, length)) + noise * noise_level

# generate sythetic data
def get_dummy_data(length):
    columns = []
    for i in range(1):
        frequency = np.random.uniform(4, 10)
        for j in range(45):
            offset = np.random.uniform(0, 2 * np.pi)
            noise_level = np.random.uniform(0.03, 0.1)
            for octave in range(1):
                columns.append(get_column(length, frequency * (2 ** octave), offset, noise_level))

    return np.array(columns).transpose()

# generate missing matrix for simulation
def get_missing_dummy_data(data):
    missing_data = data.copy()
    for i in range(int(data.shape[0])):
        size = 1
        if np.random.uniform() < 0.01:
            size = int(np.random.exponential(data.shape[0] / 100)) + 1

        col = np.random.randint(0, data.shape[1])
        row = np.random.randint(0, data.shape[0] - size)
        missing_data[row:(row + size), col] = np.NaN

    return missing_data
        
        
        
        
        
        
        
        
        
        

def main():
    data = get_dummy_data(4000)
    missing_data = get_missing_dummy_data(data)
    
    # data normalization/standardization
    print(f'Shape of missing data matrix: {missing_data.shape}')
    print(f'Average percentage of missing values [%]: {100*np.isnan(missing_data).sum()/(missing_data.shape[0]*missing_data.shape[1])}')
    
    # imputation parameters
    corr_method = 'pearson' #'pearson','spearman'
    c_thr = 0
    corr = np.abs(np.array(pd.DataFrame(missing_data).corr(method=corr_method)))
    k_range = np.array([20,10])
    batch_size = 200

    st = time.time()
    imputed = impute_wkNNr_parallel(corr, missing_data, batch_size, k_range, c_thr)
    print(f'Parallel imputation took: {time.time() - st}')
    print(f'Are there any NaN values left? {np.isnan(imputed).any()}')

    st = time.time()
    imputed = impute_wkNNr_parallel(corr, missing_data, batch_size, k_range, c_thr, parallel=False)
    print(f'Not parallel imputation took: {time.time() - st}')
    print(f'Are there any Nan values left? {np.isnan(imputed).any()}')

    inds = np.isnan(missing_data)
    for k_idx, k_neigh in enumerate(k_range):        
        print('RMSE ({} neighbors): {}'.format(k_neigh, np.sqrt(mean_squared_error(imputed[k_idx,:,:][inds], data[inds]))))
        
    # Adding first lag and lead
    shiftM1 = pd.DataFrame(missing_data).shift(periods=-1)
    shiftP1 = pd.DataFrame(missing_data).shift(periods=1)
    missing_data_add1 = pd.concat([shiftM1, pd.DataFrame(missing_data), shiftP1], axis=1)   
    corr = missing_data_add1.corr(method=corr_method).fillna(0).values
    
    print(f'Shape of missing data matrix with lag(1) lead(1): {missing_data_add1.shape}')
    
    c_start = missing_data.shape[1]
    c_end = 2 * (missing_data.shape[1])
    
    st = time.time()
    imputed = impute_wkNNr_parallel_with_add(corr, missing_data_add1.values, batch_size, k_range, c_thr, c_start, c_end)
    print(f'Parallel imputation with with lag(1) lead(1) took: {time.time() - st}')
    print(f'Are there any NaN values left? {np.isnan(imputed[:,:,c_start:c_end]).any()}')
    
    inds = np.isnan(missing_data)
    for k_idx, k_neigh in enumerate(k_range):        
        print('RMSE ({} neighbors, lag(1) lead(1)): {}'.format(k_neigh, np.sqrt(mean_squared_error(imputed[k_idx,:,c_start:c_end][inds], data[inds]))))
    
   
if __name__ == '__main__':
    main()