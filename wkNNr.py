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


def impute_wkNNr_parallel(corr, missing_data, batch_size, k_range, column_idx_to_impute, corr_thr=0, q=2, parallel=True, chunk_size=1, njobs=None):
    imputed = np.repeat(np.asarray(missing_data)[None], k_range.shape[0], axis=0)
    
    args = ((missing_data, corr, i, batch_size, k_range, corr_thr, q) for i in column_idx_to_impute)
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
    missing_data, corr, cind, batch_size, k_range, corr_thr, q = args

    start_time = time.time()
    logging.info(f'starting {cind}')
    rinds = np.isnan(missing_data[:, cind])

    # Use threshold of corr:
    correlated_columns = np.abs(corr[:, cind]) > corr_thr
    logging.info(f'{correlated_columns.sum()} correlated columns')
    filtered_cind = correlated_columns[:cind].sum()
    corr = corr[correlated_columns, :][:, correlated_columns]
    missing_data = missing_data[:, correlated_columns]

    # standardize data
    missing_data_norm = return_standardized(missing_data)
    missing_rows = missing_data_norm[rinds, :]
    present_rows = missing_data[~rinds, :]
    present_rows_norm = missing_data_norm[~rinds, :]
    present_rows_corr = present_rows_norm * corr[:, filtered_cind]
    present_rows_not_nan_mask = (~np.isnan(present_rows)).astype(int)

    total_values = np.zeros((missing_rows.shape[0], len(k_range)))
    for batch_start in range(0, missing_rows.shape[0], batch_size):
        batch = missing_rows[batch_start:batch_start + batch_size]

        weights = get_weights(batch, filtered_cind, corr, present_rows_corr, present_rows_not_nan_mask, q)

        get_values_for_weights(total_values, weights, filtered_cind, k_range, present_rows, batch_start, batch_size)
    logging.info(f'finished {cind} ({(time.time() - start_time)/60} minutes)')
    return rinds, cind, total_values


def get_weights(batch, cind, corr, present_rows_corr, present_rows_not_nan_mask, q):
    distances = nan_euclidean_distances(present_rows_corr, batch * corr[:, cind])
    counts = np.dot(present_rows_not_nan_mask, (~np.isnan(batch)).T.astype(int))
    weights = (counts**(q-0.5)) / distances #from nan_euclidean_distances we get sqrt(counts) (in addition)
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
    q = 2
    corr = pd.DataFrame(missing_data).corr(method=corr_method).fillna(0).values
    k_range = np.array([20,10])
    batch_size = 200
    column_idx_to_impute = np.arange(missing_data.shape[1])

    st = time.time()
    imputed = impute_wkNNr_parallel(corr, missing_data, batch_size, k_range, column_idx_to_impute, c_thr, q)
    print(f'\nParallel imputation took: {time.time() - st}')
    print(f'Are there any NaN values left? {np.isnan(imputed).any()}')

    st = time.time()
    imputed = impute_wkNNr_parallel(corr, missing_data, batch_size, k_range, column_idx_to_impute, c_thr, q, parallel=False)
    print(f'\nNot parallel imputation took: {time.time() - st}')
    print(f'Are there any Nan values left? {np.isnan(imputed).any()}\n')

    inds = np.isnan(missing_data)
    for k_idx, k_neigh in enumerate(k_range):        
        print('RMSE ({} neighbors): {}'.format(k_neigh, np.sqrt(mean_squared_error(imputed[k_idx,:,:][inds], data[inds]))))
        
    # Adding first lag and lead
    lag = 1
    shiftM1 = pd.DataFrame(missing_data).shift(periods=-1)
    shiftP1 = pd.DataFrame(missing_data).shift(periods=1)
    missing_data_add1 = pd.concat([shiftM1, pd.DataFrame(missing_data), shiftP1], axis=1)   
    corr = missing_data_add1.corr(method=corr_method).fillna(0).values
    
    print(f'\nShape of missing data matrix with lag(1) lead(1): {missing_data_add1.shape}')
    
    c_start = lag * missing_data.shape[1]
    c_end = (lag+1) * (missing_data.shape[1])
    column_idx_to_impute = np.arange(c_start,c_end)
    
    st = time.time()
    imputed = impute_wkNNr_parallel(corr, missing_data_add1.values, batch_size, k_range, column_idx_to_impute, c_thr, q, parallel=False)
    print(f'\nParallel imputation with lag(1) lead(1) took: {time.time() - st}')
    print(f'Are there any NaN values left? {np.isnan(imputed[:,:,c_start:c_end]).any()}\n')
    
    inds = np.isnan(missing_data)
    for k_idx, k_neigh in enumerate(k_range):        
        print('RMSE ({} neighbors, lag(1) lead(1)): {}'.format(k_neigh, np.sqrt(mean_squared_error(imputed[k_idx,:,c_start:c_end][inds], data[inds]))))
    
   
if __name__ == '__main__':
    main()
