from datasets import SplitMnist
import torch
import coreset
import vcl
import baseline
import numpy as np
from utils import plot
import time
import os

print('Running VCL')
torch.manual_seed(0)
hidden_dimensions = [256, 256]
batch_size = 512
n_epochs = 120 #TODO: Change to 120
shared_head = False
coreset_size = 0
data_class = SplitMnist()

start = time.time()
vcl_result = vcl.run_vcl(hidden_dimensions, n_epochs, data_class, coreset.random_coreset, coreset_size, batch_size, shared_head)
end = time.time()
print("VCL result: ", vcl_result)

vcl_means = [np.mean(r) for r in vcl_result]
print('Time taken for VCL: ', end-start)

#print('Running VCL with random coreset of size 40')
#torch.manual_seed(0)
#coreset_size = 40
#data_class.reset()
#random_coreset_vcl_result = vcl.run_vcl(hidden_dimensions, n_epochs, data_class, coreset.random_coreset, coreset_size, batch_size, shared_head)
#rc_vcl_means = [np.mean(r) for r in random_coreset_vcl_result]

#print('Running VCL with kcenter coreset of size 40')
#torch.manual_seed(0)
#data_class.reset()
#k_center_vcl_result = vcl.run_vcl(hidden_dimensions, n_epochs, data_class, coreset.k_center, coreset_size, batch_size, shared_head)
#kc_vcl_means = [np.mean(r) for r in k_center_vcl_result]

print("Running naive baseline")
torch.manual_seed(0)
data_class.reset()
start = time.time()
naive_baseline_result = baseline.run_baseline(hidden_dimensions, n_epochs, data_class, batch_size, shared_head)
end = time.time()
print("Naive baseline result: ", naive_baseline_result)
print('Time taken for naive baseline: ', end-start)
naive_baseline_means = [np.mean(r) for r in naive_baseline_result]

os.makedirs('results', exist_ok=True)
plot('results/split_mnist_vcl.png', [vcl_means, naive_baseline_means], ['VCL', 'Naive Baseline'])


