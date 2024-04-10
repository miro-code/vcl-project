from datasets import SplitNotMNIST, SplitMnist, PermutedMnist
import torch
import coreset
import vcl
import baseline
import pickle
import os
import sys
import numpy as np

prior_experiments_by_id = [{'prior_log_var': prior_log_var} for prior_log_var in [0.0, 0.2, 0.4, 0.6, 0.8, 1]]

def run_prior_experiment(prior_log_var):
    torch.manual_seed(0)
    dataset = 'split_notmnist'
    method = 'vcl'
    seed = 0
    torch.manual_seed(seed)
    batch_size = 512
    data_class = SplitNotMNIST()
    hidden_dimensions = [150, 150, 150, 150]
    n_epochs = 100
    shared_head = False
    result, training_times = vcl.run_vcl(hidden_dimensions, n_epochs, data_class, coreset.random_coreset, 0, batch_size, shared_head, prior_log_var=torch.tensor(prior_log_var))
    
    result_path = os.path.join('results', "prior", dataset, method, str(seed))
    os.makedirs(result_path, exist_ok=True)
    #store results and training times
    with open(os.path.join(result_path, 'result.pkl'), 'wb') as f:
        pickle.dump(result, f)
    with open(os.path.join(result_path, 'training_times.pkl'), 'wb') as f:
        pickle.dump(training_times, f)
    print("Result: ", result)
    print("Training times: ", training_times)

    #print average accuracy on last task
    print("Average accuracy on last task: ", np.mean(result[-1]))
if __name__ == '__main__':
    id = int(sys.argv[1])
    experiment = prior_experiments_by_id[id]
    prior_experiments_by_id(experiment['dataset'], experiment['method'], experiment['seed'])
    
    
