from datasets import SplitMnist
import torch
import coreset
import vcl
torch.manual_seed(0)

hidden_dimensions = [256, 256]
batch_size = None
n_epochs = 120
shared_head = False
coreset_size = 0
data_class = SplitMnist()
vcl_result = vcl.run_vcl(hidden_dimensions, n_epochs, data_class, coreset.random_coreset, coreset_size, batch_size, shared_head)
