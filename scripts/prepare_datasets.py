import gzip
import numpy as np
import os

# For pretraining we will use run 1, ckpts 3, 4, 5. Files will be downloaded as {file_type}_ {ckpt}.gz
# For probing we will use run 2, ckpt 1. Files will be downloaded as {file_type}_ 2_1.gz

data_path = '/scratch/wz1232/test_data' # Change to your data path
games = ['Amidar', 'Assault', 'Asterix', 'Boxing', 'DemonAttack', 'Frostbite', 'Gopher', 'Krull', 'Seaquest']
types = ['action', 'terminal', 'observation', 'reward']


for game in games:
    for type in types:

        # Delete the second half million of frames of each pretraining file save as numpy.
        pretrain_ckpts = [3, 4, 5]
        for ckpt in pretrain_ckpts:
            file_path = f'{data_path}/{game}/{type}_{ckpt}.gz'
            g = gzip.GzipFile(filename=file_path)
            data = np.load(g)
            data = data[:500000]

            save_path = f'{data_path}/{game}/{type}_{ckpt}.npy'
            with open(save_path, 'wb') as f:
                np.save(f, data)

            del data

        # Generate the probing training and evaluation datasets and save as numpy.
        probe_ckpts = [1,50]
        for ckpt in probe_ckpts:
            file_path = f'{data_path}/{game}/{type}_2_{ckpt}.gz'
            g = gzip.GzipFile(filename=file_path)
            data = np.load(g)

            ft_save_path = f'{data_path}/{game}/{type}_2_{ckpt}_ft.npy'
            eval_save_path = f'{data_path}/{game}/{type}_2_{ckpt}_eval.npy'

            with open(eval_save_path, 'wb') as f:
                np.save(f, data[:20000])

            with open(ft_save_path, 'wb') as f:
                np.save(f, data[20000:100000])

            del data

