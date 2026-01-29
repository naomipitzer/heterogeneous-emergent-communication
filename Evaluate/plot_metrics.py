#------------ Imports ------------#
import matplotlib.pyplot as plt

# Function to plot Training vs. Validation Accuracy
def plot_accuracy(training_accuracies, validation_accuracies, save_path='accuracy_plot.png'):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(training_accuracies) + 1)
    
    plt.plot(epochs, training_accuracies, label='Training Accuracy')
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs. Validation Accuracy')

    # Remove this line to let matplotlib decide the best ticks
    # plt.xticks(epochs)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


# Function to plot Training vs. Validation Loss
def plot_loss(training_losses, validation_losses, save_path='loss_plot.png'):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(training_losses) + 1)
    plt.plot(epochs, training_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs. Validation Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()


# Function to plot Learning Rate vs. Loss
def plot_lr_vs_loss(learning_rates, losses, save_path='lr_vs_loss.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, losses, label='Learning Rate vs Loss')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate vs. Loss')
    plt.xscale('log')
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def plot_entropy(receiver_entropies, sender_entropies, save_path='entropy_plot.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(receiver_entropies, label='Receiver Entropy')
    plt.plot(sender_entropies, label='Sender Entropy')
    plt.xlabel('Epoch')
    plt.ylabel('Entropy')
    plt.title('Receiver vs Sender Entropy')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()
    
# Function to plot Time Taken Per Epoch
def plot_epoch_times(epoch_times, save_path='epoch_times.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_times, label='Epoch Time (s)')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Time Taken Per Epoch')
    plt.grid()
    plt.savefig(save_path)
    plt.close()

# Function to plot CPU and GPU Usage
def plot_system_usage(cpu_usage, gpu_memory_usage, epoch_list, save_path='system_usage.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, cpu_usage, label='CPU Usage (%)')
    plt.plot(epoch_list, gpu_memory_usage, label='GPU Memory Usage (GB)')
    plt.xlabel('Epoch')
    plt.ylabel('Usage')
    plt.title('System Resource Usage')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()
