from datasets import SplitMnist
import torch
import coreset
import vcl
import numpy as np
from utils import plot

print('Running VCL')
torch.manual_seed(0)
hidden_dimensions = [256, 256]
batch_size = 512
n_epochs = 120 #TODO: Change to 120
shared_head = False
coreset_size = 0
data_class = SplitMnist()
vcl_result = vcl.run_vcl(hidden_dimensions, n_epochs, data_class, coreset.random_coreset, coreset_size, batch_size, shared_head)
vcl_means = [np.mean(r) for r in vcl_result]

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
naive_baseline_result = vcl.run_baseline(hidden_dimensions, n_epochs, data_class, coreset.random_coreset, coreset_size, batch_size, shared_head)
naive_baseline_means = [np.mean(r) for r in naive_baseline_result]




plot('split_mnist_vcl.png', vcl_means, naive_baseline_means, naive_baseline_means)


