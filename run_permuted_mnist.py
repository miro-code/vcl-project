from datasets import PermutedMnist
import torch
import coreset
import vcl
import baseline
from utils import plot
import time
import os
import numpy as np


hidden_size = [100, 100]
batch_size = 256
n_epochs = 2
shared_head = True
n_tasks = 5



torch.manual_seed(0)
coreset_size = 0
data_gen = PermutedMnist(n_tasks)
vcl_result = vcl.run_vcl(hidden_size, n_epochs, data_gen, coreset.random_coreset, coreset_size, batch_size, shared_head)
print(vcl_result)
vcl_mean = [np.mean(r) for r in vcl_result]

torch.manual_seed(0)
coreset_size = 200
data_gen = PermutedMnist(n_tasks)
random_coreset_result = vcl.run_vcl(hidden_size, n_epochs, data_gen,coreset.random_coreset, coreset_size, batch_size, shared_head)
print(random_coreset_result)
rc_vcl_mean = [np.mean(r) for r in random_coreset_result]



data_gen = PermutedMnist(n_tasks)
k_center_coreset_result = vcl.run_vcl(hidden_size, n_epochs, data_gen, coreset.k_center, coreset_size, batch_size, shared_head)
print(k_center_coreset_result)
kc_vcl_mean = [np.mean(r) for r in k_center_coreset_result]

os.makedirs('results', exist_ok=True)
plot('results/permuted.jpg', vcl_mean, rc_vcl_mean, kc_vcl_mean, 'VCL', 'Random Coreset VCL', 'K Center Coreset VCL')