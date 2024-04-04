import numpy as np
import torch
import utils
from models import MLP, BNN

def run_vcl(hidden_dims, n_epochs, data_class, coreset_func, coreset_size=0, batch_size=264, single_head=True):
    input_dim, out_dim = data_class.get_dims()
    task_accuracies = np.array([])
    X_test_by_task, t_test_by_task = [], []
    X_coresets, y_coresets = [], []
    
    for task_id in range(data_class.n_tasks):
        X_train, y_train, X_test, y_test = data_class.next_task()
        X_test_by_task.append(X_test)
        t_test_by_task.append(y_test)

        head = 0 if single_head else task_id

        # Train first network with maximum likelihood=SGD (It seems strange not to use VI here but it is what the original code does)
        if task_id == 0:
            ml_model = MLP(input_dim, hidden_dims, out_dim, X_train.shape[0])
            ml_model.train(X_train, y_train, task_id, n_epochs, batch_size)
            mf_weights = ml_model.get_weights()
            mf_variances = None
            ml_model.close_session()
        if coreset_size > 0:
            X_coresets, y_coresets, X_train, y_train = coreset_func(X_coresets, y_coresets, X_train, y_train, coreset_size)
        mf_model = BNN(input_dim, hidden_dims, out_dim, X_train.shape[0], prev_means=mf_weights, prev_log_variances=mf_variances)
        mf_model.train(X_train, y_train, head, n_epochs, batch_size)
        mf_weights, mf_variances = mf_model.get_weights()

        # Incorporate coreset data and make prediction
        accuracy = utils.get_scores(mf_model, X_test_by_task, t_test_by_task, X_coresets, y_coresets, hidden_dims, n_epochs, single_head, batch_size)
        task_accuracies = utils.concatenate_results(accuracy, task_accuracies)

        mf_model.close_session()

    return task_accuracies
