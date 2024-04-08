import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')



def load_result(dataset, method, seed):
    result_path = os.path.join('results', dataset, method, str(seed))
    with open(os.path.join(result_path, 'result.pkl'), 'rb') as f:
        result = pickle.load(f)
    with open(os.path.join(result_path, 'training_times.pkl'), 'rb') as f:
        training_times = pickle.load(f)
    return result, training_times

def plot_general(x, ys, errors, labels, title, xlabel, ylabel, filename):
    plt.figure()
    for y, error, label in zip(ys, errors, labels):
        plt.errorbar(x, y, yerr=error, label=label, fmt='o', capsize=5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()  


def plot_average_accuracy(dataset, methods):
    avg_accuracies = []
    accuracies_std = []
    labels = []
    for method in methods:
        all_accuracies = []
        for seed in range(5):
            result, _ = load_result(dataset, method, seed)
            accuracies = [np.mean(r) for r in result]
            all_accuracies.append(accuracies)
        avg_accuracies.append(np.mean(all_accuracies, axis=0))
        accuracies_std.append(np.std(all_accuracies, axis=0))
        labels.append(method)
    
    tasks = range(1, len(avg_accuracies[0]) + 1)  # Assuming tasks are indexed from 1
    filename = f"results/{dataset}_average_accuracy.png"
    plot_general(tasks, avg_accuracies, accuracies_std, labels, f'Average Accuracy - {dataset}', 'Task', 'Average Accuracy', filename)

def plot_task_accuracy(dataset, methods, task_index):
    accuracies_by_method = []
    accuracies_std_by_method = []
    labels = []
    
    for method in methods:
        task_accuracies_across_seeds = []
        
        for seed in range(5):
            result, _ = load_result(dataset, method, seed)
            task_accuracies = [step_acc[task_index] if len(step_acc) > task_index else np.nan for step_acc in result]
            task_accuracies_across_seeds.append(task_accuracies)
        
        mean_accuracies = [np.nanmean(step) if step is not None else None for step in task_accuracies_across_seeds]
        stddev_accuracies = [np.nanstd(step) if step is not None else None for step in task_accuracies_across_seeds]
        
        accuracies_by_method.append(mean_accuracies)
        accuracies_std_by_method.append(stddev_accuracies)
        labels.append(method)
    
    steps = range(1, len(mean_accuracies) + 1)
    filename = f"results/{dataset}_task_{task_index+1}_accuracy.png" 
    plot_general(steps, accuracies_by_method, accuracies_std_by_method, labels, f'Accuracy for Task {task_index+1} - {dataset}', 'Learning Step', 'Accuracy', filename)

def plot_cumulative_training_time(dataset, methods):
    cumulative_times_by_method = []
    labels = []
    
    for method in methods:
        cumulative_times_across_seeds = []
        for seed in range(5):
            _, training_times = load_result(dataset, method, seed)
            cumulative_times = np.cumsum(training_times)
            cumulative_times_across_seeds.append(cumulative_times)
        mean_cumulative_times = np.mean(cumulative_times_across_seeds, axis=0)
        stddev_cumulative_times = np.std(cumulative_times_across_seeds, axis=0)
        cumulative_times_by_method.append(mean_cumulative_times)
        labels.append(method)
    tasks = range(1, len(mean_cumulative_times) + 1)
    filename = f"results/{dataset}_cumulative_training_time.png"
    plot_general(tasks, cumulative_times_by_method, [stddev_cumulative_times]*len(methods), labels, f'Cumulative Training Time - {dataset}', 'Task', 'Cumulative Time (s)', filename)


if __name__ == '__main__':
    datasets = ['split_notmnist', 'split_mnist', 'permuted_mnist'] 
    methods = ['vcl', 'random_coreset', 'k_center', 'baseline']
    for dataset in datasets:
        plot_average_accuracy(dataset, methods)
        plot_cumulative_training_time(dataset, methods)
        for i in range(5):
            plot_task_accuracy(dataset, methods, i)