import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from models import BNN

matplotlib.use('Agg')


def plot(filename, records, labels):

    fig = plt.figure(figsize=(7,3))
    ax = plt.gca()
    for record,label in zip(records, labels):
        plt.plot(np.arange(len(record))+1, record, label=label, marker='o')
    ax.set_xticks(range(1, len(records[0])+1))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('\# tasks')
    ax.legend()

    fig.savefig(filename, bbox_inches='tight')
    plt.close()