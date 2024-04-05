import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from models import BNN

matplotlib.use('Agg')


def plot(filename, vcl, random_coreset_vcl, k_center_vcl):

    fig = plt.figure(figsize=(7,3))
    ax = plt.gca()
    plt.plot(np.arange(len(vcl))+1, vcl, label='VCL', marker='o')
    plt.plot(np.arange(len(random_coreset_vcl))+1, random_coreset_vcl, label='VCL + Random Coreset', marker='o')
    plt.plot(np.arange(len(k_center_vcl))+1, k_center_vcl, label='VCL + K-center Coreset', marker='o')
    ax.set_xticks(range(1, len(vcl)+1))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('\# tasks')
    ax.legend()

    fig.savefig(filename, bbox_inches='tight')
    plt.close()