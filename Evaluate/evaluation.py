import matplotlib.pyplot as plt
import numpy as np
import math


# Prints list class label, average conversation length, variance
# Exports PNG bar plot of this. 
def track_conversation_length(conv_lengths):
    """
    Takes {class_label: [lengths]} and prints stats + creates a bar plot.
    Saves plot as 'conversation_lengths.png'.
    """
    class_ids = sorted(conv_lengths.keys())
    means = []
    variances = []

    for cid in class_ids:
        lengths = conv_lengths[cid]
        mean_len = np.mean(lengths)
        var_len = np.var(lengths)
        means.append(mean_len)
        variances.append(var_len)
        #print(f"Class {cid}\nMean: {mean_len}, Variance: {var_len:.2f}\n")

    # Plot
    x_pos = np.arange(len(class_ids))
    fig, ax = plt.subplots()
    ax.bar(x_pos, means, yerr=variances, align='center', alpha=0.7, ecolor='black', capsize=10)
    ax.set_ylabel('Conversation Length')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Class {i}' for i in class_ids])
    ax.set_title('Average Conversation Lengths per Class')
    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.savefig('conversation_lengths.png')
    plt.close()



def convmean_per_class(conv_lengths):
    """
    Takes {class_label: [lengths]}, prints stats, and returns a dict of mean lengths per class.
    """
    class_ids = sorted(conv_lengths.keys())
    mean_per_class = {}

    for cid in class_ids:
        lengths = conv_lengths[cid]
        mean_len = np.mean(lengths)
        var_len = np.var(lengths)
        mean_per_class[cid] = mean_len
        #print(f"Class {cid}\nMean: {mean_len:.2f}, Variance: {var_len:.2f}\n")

    return mean_per_class


def lengths_over_time(conv_lengths_epochs):
    """
    Plots how average conversation lengths per class change over epochs.
    Saves plot as 'conversation_lengths_over_epochs.png'.
    """
    if not conv_lengths_epochs:
        print("No data to plot.")
        return

    classes = sorted(conv_lengths_epochs[0].keys())
    epochs = list(range(1, len(conv_lengths_epochs) + 1))

    for cid in classes:
        class_means = [epoch_data.get(cid, 0) for epoch_data in conv_lengths_epochs]
        plt.plot(epochs, class_means, label=f'Class {cid}')

    plt.xlabel('Epoch')
    plt.ylabel('Average Conversation Length')
    plt.title('Conversation Lengths Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('conversation_lengths_over_epochs.png')
    plt.close()
    print("Saved conversation_lengths_over_epochs.png")
