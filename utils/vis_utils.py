"""
Define some utility functions
"""
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from six.moves import cPickle as pickle

def snapshot_experiment_eval(logdir, experiment_id, data):
    """
    Store the output of the experiment in a file
    """
    snapshot_file = logdir + '/' + experiment_id + '.pickle'
    with open(snapshot_file, 'wb') as f:
        pickle.dump(data, f)

    print('Experimental Eval has been snapshotted to %s!'%(snapshot_file))

def snapshot_task_labels(logdir, experiment_id, data):
    """
    Store the output of the experiment in a file
    """
    snapshot_file = logdir + '/' + experiment_id + '_task_labels.pickle'
    with open(snapshot_file, 'wb') as f:
        pickle.dump(data, f)

    print('Experimental Eval has been snapshotted to %s!'%(snapshot_file))

def snapshot_experiment_meta_data(logdir, experiment_id, exper_meta_data):
    """
    Store the meta-data of the experiment in a file
    """
    meta_file = logdir + '/' + experiment_id + '.txt'
    with open(meta_file, 'wb') as f:
        for key in exper_meta_data:
            print('{}: {}'.format(key, exper_meta_data[key]))
            f.write('{}:{} \n'.format(key, exper_meta_data[key]))

    print('Experimental meta-data has been snapshotted to %s!'%(meta_file))

def plot_acc_multiple_runs(data, task_labels, valid_measures, n_stats, plot_name=None):
    """
    Plots the accuracies
    Args:
        task_labels List of tasks
        n_stats     Number of runs
        plot_name   Name of the file where the plot will be saved

    Returns:
    """
    n_tasks = len(task_labels)
    plt.figure(figsize=(14, 3))
    axs = [plt.subplot(1,n_tasks+1,1)]
    for i in range(1, n_tasks + 1):
        axs.append(plt.subplot(1, n_tasks+1, i+1, sharex=axs[0], sharey=axs[0]))

    fmt_chars = ['o', 's', 'd']
    fmts = []
    for i in range(len(valid_measures)):
        fmts.append(fmt_chars[i%len(fmt_chars)])

    plot_keys = sorted(data['mean'].keys())

    for k, cval in enumerate(plot_keys):
        label = "c=%g"%cval
        mean_vals = data['mean'][cval]
        std_vals = data['std'][cval]
        for j in range(n_tasks+1):
            plt.sca(axs[j])
            errorbar_kwargs = dict(fmt="%s-"%fmts[k], markersize=5)
            if j < n_tasks:
                norm= np.sqrt(n_stats) # np.sqrt(n_stats) for SEM or 1 for STDEV
                axs[j].errorbar(np.arange(n_tasks)+1, mean_vals[:, j], yerr=std_vals[:, j]/norm, label=label, **errorbar_kwargs)
            else:
                mean_stuff = []
                std_stuff = []
                for i in range(len(data['mean'][cval])):
                    mean_stuff.append(data['mean'][cval][i][:i+1].mean())
                    std_stuff.append(np.sqrt((data['std'][cval][i][:i+1]**2).sum())/(n_stats*np.sqrt(n_stats)))
                plt.errorbar(range(1,n_tasks+1), mean_stuff, yerr=std_stuff, label="%s"%valid_measures[k], **errorbar_kwargs)
            plt.xticks(np.arange(n_tasks)+1)
            plt.xlim((1.0,5.5))
            """
            # Uncomment this if clutter along y-axis needs to be removed
            if j == 0:
                axs[j].set_yticks([0.5,1])
            else:
                plt.setp(axs[j].get_yticklabels(), visible=False)
            plt.ylim((0.45,1.1))
            """

    for i, ax in enumerate(axs):
        if i < n_tasks:
            ax.set_title((['Task %d (%d to %d)'%(j+1,task_labels[j][0], task_labels[j][-1])\
                           for j in range(n_tasks)] + ['average'])[i], fontsize=8)
        else:
            ax.set_title("Average", fontsize=8)
        ax.axhline(0.5, color='k', linestyle=':', label="chance", zorder=0)

    handles, labels = axs[-1].get_legend_handles_labels()

    # Reorder legend so chance is last
    axs[-1].legend([handles[j] for j in [i for i in range(len(valid_measures)+1)]], 
                    [labels[j] for j in [i for i in range(len(valid_measures)+1)]], loc='best', fontsize=6)

    axs[0].set_xlabel("Tasks")
    axs[0].set_ylabel("Accuracy")
    plt.gcf().tight_layout()
    plt.grid('on')
    if plot_name == None:
        plt.show()
    else:
        plt.savefig(plot_name)

def plot_histogram(data, n_bins=10, plot_name='my_hist'):
    plt.hist(data, bins=n_bins)
    plt.savefig(plot_name)
    plt.close()
