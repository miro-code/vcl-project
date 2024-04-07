import numpy as np
import torch
from models import MLP, BNN


def get_results(model, X_test_by_task, y_test_by_task, X_cs_by_task, y_cs_by_task, hidden_size, n_epochs, shared_head, batch_size):
    #takes the model, trains it on the coreset data and returns the accuracy

    parameter_means, parameter_variances = model.get_parameters()
    
    accuracies = []

    if shared_head:
        if X_cs_by_task:
            X_train = torch.cat(X_cs_by_task, dim=0)
            y_train = torch.cat(y_cs_by_task, dim=0)
            #TODO: Does this mean that when training on the coreset data, each task gets different backbone weights?
            #It seems information on which task the data belongs to is also passed to the model
            final_model = BNN(X_train.shape[1], hidden_size, y_train.shape[1], X_train.shape[0], previous_means=parameter_means, previous_log_variances=parameter_variances)
            final_model.train(X_train, y_train, 0, n_epochs, batch_size)
        else:
            final_model = model
    else:
        final_models = []
        for i in range(len(X_test_by_task)):
            if X_cs_by_task:
                X_train, y_train = X_cs_by_task[i], y_cs_by_task[i]
                final_model = BNN(X_train.shape[1], hidden_size, y_train.shape[1], X_train.shape[0], previous_means=parameter_means, previous_log_variances=parameter_variances)
                final_model.train(X_train, y_train, i, n_epochs, batch_size)
            else:
                final_model = model
            final_models.append(final_model)
    
    for i in range(len(X_test_by_task)):
        final_model = final_models[i] if not shared_head else final_model
        head = 0 if shared_head else i
        x_test, y_test = X_test_by_task[i], y_test_by_task[i]
        predictions = final_model.predict_proba(x_test, head)
        predictions = torch.argmax(predictions, axis=1)
        y = torch.argmax(y_test, axis=1)
        accuracy = torch.sum(predictions == y).item() / y.shape[0]
        accuracies.append(accuracy)
    return accuracies

def run_vcl(hidden_dims, n_epochs, data_class, coreset_func, coreset_size=0, batch_size=264, shared_head=True):
    input_dim, out_dim = data_class.get_dims()
    task_accuracies = []
    X_test_by_task, y_test_by_task = [], []
    X_cs_by_task, y_cs_by_task = [], []
    
    for task_id in range(data_class.n_tasks):
        print('Task ', task_id)
        X_train, y_train, X_test, y_test = data_class.next_task()
        X_test_by_task.append(X_test)
        y_test_by_task.append(y_test)

        head = 0 if shared_head else task_id

        # Train first network with maximum likelihood=SGD (It seems strange but it is what the original code does)
        if task_id == 0:
            mlp = MLP(input_dim, hidden_dims, out_dim, X_train.shape[0])
            mlp.train(X_train, y_train, task_id, n_epochs, batch_size)
            parameter_means, parameter_variances = mlp.get_parameters()
        
        if coreset_size > 0:
            X_cs_by_task, y_cs_by_task, X_train, y_train = coreset_func(X_cs_by_task, y_cs_by_task, X_train, y_train, coreset_size)
        bnn = BNN(input_dim, hidden_dims, out_dim, X_train.shape[0], previous_means=parameter_means, previous_log_variances=parameter_variances)
        #TODO:remove debugging
        bnn.X_test = X_test
        bnn.y_test = y_test
        #TODO:remove debugging
        
        bnn.train(X_train, y_train, head, n_epochs, batch_size)
        parameter_means, parameter_variances = bnn.get_parameters()

        # Incorporate coreset data and make prediction
        accuracy = get_results(bnn, X_test_by_task, y_test_by_task, X_cs_by_task, y_cs_by_task, hidden_dims, n_epochs, shared_head, batch_size)
        task_accuracies.append(accuracy)
    return task_accuracies
