from datasets import SplitMnist
import torch
import coreset
import vcl


hidden_dims = [256, 256]
batch_size = None
no_epochs = 120
single_head = False

# Run vanilla VCL
torch.manual_seed(0)

coreset_size = 0
data_gen = SplitMnist()
vcl_result = vcl.run_vcl(hidden_dims, no_epochs, data_gen, 
    coreset.rand_from_batch, coreset_size, batch_size, single_head)
