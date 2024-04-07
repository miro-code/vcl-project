from datasets import SplitMnist
import torch
import coreset
import vcl
import baseline
from utils import plot
import time
import os
import numpy as np

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
print('VCL means: ', vcl_means)
print('Time taken for VCL: ', end-start)

print('Running VCL with random coreset of size 40')
torch.manual_seed(0)
coreset_size = 40
data_class.reset()
start = time.time()
random_coreset_vcl_result = vcl.run_vcl(hidden_dimensions, n_epochs, data_class, coreset.random_coreset, coreset_size, batch_size, shared_head)
end = time.time()
print("Random coreset VCL result: ", random_coreset_vcl_result)
rc_vcl_means = [np.mean(r) for r in random_coreset_vcl_result]
print('Random coreset VCL means: ', rc_vcl_means)
print('Time taken for random coreset VCL: ', end-start)



print('Running VCL with kcenter coreset of size 40')
torch.manual_seed(0)
data_class.reset()
start = time.time()
k_center_vcl_result = vcl.run_vcl(hidden_dimensions, n_epochs, data_class, coreset.k_center, coreset_size, batch_size, shared_head)
end = time.time()
print("K center VCL result: ", k_center_vcl_result)
kc_vcl_means = [np.mean(r) for r in k_center_vcl_result]
print('K center VCL means: ', kc_vcl_means)
print('Time taken for k center VCL: ', end-start)

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
plot('results/split_mnist_vcl.png', [vcl_means, naive_baseline_means, kc_vcl_means, rc_vcl_means], ['VCL', 'Naive Baseline', 'K Center VCL', 'Random Coreset VCL'])


