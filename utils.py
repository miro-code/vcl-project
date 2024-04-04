import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from models import BNN

matplotlib.use('agg')

def plot(filename, vcl, rand_vcl, kcen_vcl):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig = plt.figure(figsize=(7,3))
    ax = plt.gca()
    plt.plot(np.arange(len(vcl))+1, vcl, label='VCL', marker='o')
    plt.plot(np.arange(len(rand_vcl))+1, rand_vcl, label='VCL + Random Coreset', marker='o')
    plt.plot(np.arange(len(kcen_vcl))+1, kcen_vcl, label='VCL + K-center Coreset', marker='o')
    ax.set_xticks(range(1, len(vcl)+1))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('\# tasks')
    ax.legend()

    fig.savefig(filename, bbox_inches='tight')
    plt.close()