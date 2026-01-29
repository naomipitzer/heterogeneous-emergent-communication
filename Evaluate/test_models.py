#------------------------------------------Imports------------------------------------------#

# Basic Model and Training Stuff
from archs import Sender, Receiver, Baseline 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset,random_split
import numpy as np
import random
from torch.autograd import Variable
from typing import Dict, Any, Tuple, List
from torch import Tensor
from collections import defaultdict
import matplotlib.pyplot as plt

#Evaluation
from evaluation import track_conversation_length, convmean_per_class, lengths_over_time

# System Measurement - Time, RAM, CPU etc.
import psutil
import time
from tqdm import tqdm
from plot_metrics import plot_accuracy, plot_loss, plot_lr_vs_loss, plot_epoch_times, plot_system_usage, plot_entropy

#Data Libraries
import pickle
import os
import glob
from PIL import Image
import torchaudio
import torchaudio.transforms as audioT
import torchvision.transforms as imageT

# Embedding Generator Models
from torchvggish import vggish, vggish_input
from torchvision import models





#------------------------------------------Parameter Definition------------------------------------------#
device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
print("Device: "+str(device)+"\n")
conv_lengths_epochs = []
runfile = 'ms-size-50-uni/'

#-------------Game Settings--------------#
unimodal = False
audio = True # if audio is sender
dynamic = True
sender_is_learning = True # Funny that this actually came from a bug

n_classes = 6
n_distractors = 2
message_size = 50
max_conv_length=10
audio_ft_size = 128 # Changeable: image=128, audio=whateve
image_ft_size = 128 
fixed_exchange = False
batch_size=1

sender_hidden_size = 128
receiver_hidden_size = 128

# Instantiating Models. If audio is sender, sender audio. If unimodal etc
if audio:
    sender = Sender(feat_dim=audio_ft_size, h_dim=sender_hidden_size, w_dim=message_size, bin_dim_out=message_size, use_binary=True).to(device)
    if unimodal:
        receiver = Receiver(z_dim=message_size, desc_dim=audio_ft_size, hid_dim=receiver_hidden_size, out_dim=1, w_dim=message_size, s_dim=1, use_binary=True).to(device)
    else:
        receiver = Receiver(z_dim=message_size, desc_dim=image_ft_size, hid_dim=receiver_hidden_size, out_dim=1, w_dim=message_size, s_dim=1, use_binary=True).to(device)
        
else:
    sender = Sender(feat_dim=image_ft_size, h_dim=sender_hidden_size, w_dim=message_size, bin_dim_out=message_size, use_binary=True).to(device)
    if unimodal:
        receiver = Receiver(z_dim=message_size, desc_dim=image_ft_size, hid_dim=receiver_hidden_size, out_dim=1, w_dim=message_size, s_dim=1, use_binary=True).to(device)
    else: 
        receiver = Receiver(z_dim=message_size, desc_dim=audio_ft_size, hid_dim=receiver_hidden_size, out_dim=1, w_dim=message_size, s_dim=1, use_binary=True).to(device)

baseline_sen = Baseline(hid_dim=128, x_dim=128, binary_dim=message_size, inp_dim=0).to(device)
baseline_rec = Baseline(hid_dim=128, x_dim=128, binary_dim=message_size, inp_dim=0).to(device)

print(f"Game settings:\nDistractors={n_distractors}   Classes={n_classes}  MessageSize={message_size}   AudioSize={audio_ft_size}   ImageSize={image_ft_size}\n")




#----------------------- Load up models and data ----------------------–#
# How to load them up later:
checkpoint = torch.load(runfile+'models_checkpoint.pth')

sender.load_state_dict(checkpoint['sender_state_dict'])
receiver.load_state_dict(checkpoint['receiver_state_dict'])
baseline_sen.load_state_dict(checkpoint['baseline_sen_state_dict'])
baseline_rec.load_state_dict(checkpoint['baseline_rec_state_dict'])
sender.eval()
receiver.eval()


class SyntheticData(Dataset):
    def __init__(self, n_distractors,audio_embedding_file,image_embedding_file,transform=None):
        """
        - audio_root_dir: Path to `synthetic_audio/`
        - image_root_dir: Path to `synthetic_shapes/`
        - transform: Any optional transformations (not used here)
        """
        self.transform = transform
        self.n_distractors = n_distractors

        if audio:
            # Load audio data
            data = np.load(audio_embedding_file)
            dis_data = np.load(audio_embedding_file if unimodal else image_embedding_file)
        else:
            #Load images
            data = np.load(image_embedding_file)
            dis_data = np.load(image_embedding_file if unimodal else audio_embedding_file)


        #Sender's input
        self.embeddings = data['embeddings']
        self.labels = data['labels']

        # Receiver's input
        self.distembeddings = dis_data['embeddings']
        self.distlabels = dis_data['labels']

        

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        """
        Returns:
        - Embedding (Tensor) of shape [N * 128]
        - Class label (int)
        """
        target_embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        label = int(self.labels[idx])

        dist_embeddings = []
        if not dynamic:
            dist_embeddings = emb_set
        else:
            for i in range(0,6):
                idx = np.where(self.distlabels == i)[0]
                random_index = np.random.choice(idx)
                emb = torch.tensor(self.distembeddings[random_index], dtype=torch.float32)
                dist_embeddings.append(emb)
            dist_embeddings = torch.stack(dist_embeddings, dim=0).squeeze(1)
            
            
        correct_image_index = label
        
        
        return target_embedding.to(device), dist_embeddings.to(device), torch.tensor(label, dtype=torch.long).to(device), torch.tensor(correct_image_index, dtype=torch.long).to(device)
# Loading it up later
with open(runfile+'test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



#------------------------------------------Conversation Function------------------------------------------#
def conversation(sender, receiver, baseline_sen, baseline_rec, exchange_args: Dict[str, Any]):
    """
    Handles a communication exchange between a sender and a receiver.
    The sender encodes an audio feature into a message, which is interpreted by the receiver.
    
    Parameters:
        sender: Sender model that generates communication signals.
        receiver: Receiver model that interprets messages from the sender.
        baseline_sen: Baseline sender model used for scoring.
        baseline_rec: Baseline receiver model used for scoring.
        exchange_args: Dictionary containing all input arguments.

    Returns:
        Tuple containing conversation details including stop conditions, messages, predictions, and losses.
    """
    
    # Extract conversation parameters
    audio_feats = exchange_args["audio"]  # Audio features input
    target = exchange_args["target"]  # Class labels
    distractors = exchange_args["distractors"]  # Set of distractors
    train = exchange_args["train"]  # True for training, False for evaluation
    break_early = exchange_args.get("break_early", False)  # Whether to terminate early

    batch_size = audio_feats.size(0) # If one audio clip, then batch size = 1
    first_rec = torch.zeros(sender.w_dim).to(device)  # Placeholder for first message initialization (starts at 00000... etc)

    # Initialize communication tracking variables
    stop_mask = [torch.ones(batch_size, 1, dtype=torch.uint8)]  # Mask for stopping conversation
    stop_feat, stop_prob = [], []  # Stop features and probabilities
    sen_feats, sen_probs = [], []  # Sender messages and probabilities
    rec_feats, rec_probs = [], []  # Receiver messages and probabilities
    y, bs, br = [], [], []  # Predictions, sender loss, receiver loss

    # Define the initial binary message
    w_binary = first_rec.expand(batch_size, sender.w_dim).clone().to(device) # Cloning this into a diff memory address

    # Set training mode
    if train:
        sender.train()
        receiver.train()
        baseline_sen.train()
        baseline_rec.train()
    else:
        sender.eval()
        receiver.eval()

    receiver.reset_state()  # Reset receiver state before communication begins

    max_exchange = max_conv_length  # Define maximum message exchanges to prevent infinite loops

    #---------Conversation Loop----------#
    #print("--------------------Conversation-------------------")
    for i_exch in range(max_exchange):
        z_r = w_binary.to(device)  # Receiver's message
        #print("Receiver: "+str(z_r))
        
        # Sender processes audio features and previous message to generate new communication
        with torch.no_grad() if not train else torch.enable_grad():
            z_binary, z_probs = sender(audio_feats, z_r)

    
        z_s = z_binary  # Sender's message to be received
        
        # Receiver interprets the sender's message
        with torch.no_grad() if not train else torch.enable_grad():
            (s_binary, s_prob), (w_binary, w_probs), outp = receiver(z_s, exchange_args["distractors"])
        
        # Compute baseline scores if training
        if train:
            sen_h_x: Tensor = sender.h_x.to(device) # Sender hidden states
            with torch.no_grad():
                baseline_sen_scores = baseline_sen(sen_h_x, z_r, None) # Estimates loss using senders internal state + receiver's message

            rec_h_z: Tensor = receiver.h_z if receiver.h_z.to(device) is not None else receiver.initial_state(batch_size).to(device) #Receiver's hidden state (if None initializes new state)
            with torch.no_grad():
                #print("Message size: "+str(z_s.shape) + " Hidden State size: "+str(rec_h_z.shape))
                baseline_rec_scores = baseline_rec(None, z_s, rec_h_z) #Estimates scores using sender's msg and receiver's hidden state

        # Compute log probabilities and determine predictions
        outp = outp.view(batch_size, -1)
        dist = F.log_softmax(outp, dim=1)
        maxdist, argmax = dist.max(dim=1) # Model's final prediction, never used?

        # Store conversation history
        stop_mask.append(torch.min(stop_mask[-1], s_binary.byte()))
        stop_feat.append(s_binary)
        stop_prob.append(s_prob)
        sen_feats.append(z_binary)
        sen_probs.append(z_probs)
        rec_feats.append(w_binary)
        rec_probs.append(w_probs)
        y.append(outp)

        if train:
            br.append(baseline_rec_scores)
            bs.append(baseline_sen_scores)

        # Terminate exchange if all conversations are complete
        if break_early and stop_mask[-1].float().sum().item() == 0:
            break

    # Ensure final stop mask is zero
    stop_mask[-1].fill_(0)
    
    # Return results of conversation
    s = (stop_mask, stop_feat, stop_prob)
    sen_w = (sen_feats, sen_probs)
    rec_w = (rec_feats, rec_probs)
    #print("------------------Conversation Done------------------")
    #print(y)
    
    return s, sen_w, rec_w, y, bs, br



def entropy_per_conv(samples=8):
    num_batches = len(test_loader)
    selected_indices = set(random.sample(range(num_batches), samples))

    entropy_per = []
    for idx, batch in enumerate(test_loader):
        if idx in selected_indices:
            #print(f"Batch {idx}: Selected for special processing")
            audio, distractors, target, correct_index = batch
            exchange_args = {
                    "audio": audio,  
                    "target": target,
                    "distractors": distractors,
                    "desc": None,  
                    "train": False,  # Set to False for testing
                    "break_early": False
            }

            s, sen_w, rec_w, y, bs, br = conversation(sender, receiver, baseline_sen, baseline_rec, exchange_args)
            s_masks, s_feats, s_probs = s
                
            # for i, tensor in enumerate(s_feats):
            # # Assuming each tensor is at least of size [1] or more
            #     first_value = tensor[0]
            #     print(f"Tensor {i} - First value: {first_value}")
            
            sen_feats, sen_probs = sen_w
            rec_feats, rec_probs = rec_w
        
            # Mask loss if dynamic exchange length
            binary_s_masks = binary_rec_masks = binary_sen_masks = None
            bas_rec_masks = bas_sen_masks = None
            y_masks = None

            classification_entropy = []
            for timestep in y:
                dist = F.log_softmax(timestep, dim=1)
                entropy_vals = entropy_from_log_probs(dist)

                entropy_scalar = entropy_vals[0].squeeze().item()
                classification_entropy.append(entropy_scalar)
        
            entropy_per.append(classification_entropy)

            timesteps = list(range(1, 11))  # Assuming 10 timesteps

            plt.figure(figsize=(10, 6))
            for i, entropy_list in enumerate(entropy_per):
                plt.plot(timesteps, entropy_list, label=f'Sample {i+1}')
            
            plt.xlabel('Timestep')
            plt.ylabel('Classification Entropy')
            plt.title('Entropy per Conversation Over Time')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('entropy_per_conversation_timestep.png')
            plt.close()
            print("Saved plot as entropy_per_conversation.png")



from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

def f1_vs_convlength():
    conv_lengths = defaultdict(list)
    all_true = []
    all_pred = []

    for idx, batch in enumerate(test_loader):
        audio, distractors, target, correct_index = batch
        exchange_args = {
            "audio": audio,  
            "target": target,
            "distractors": distractors,
            "desc": None,  
            "train": False,
            "break_early": False
        }

        s, sen_w, rec_w, y, bs, br = conversation(sender, receiver, baseline_sen, baseline_rec, exchange_args)
        s_masks, s_feats, s_probs = s

        # Get predictions
        outp, ent_y_rec = get_rec_outp(y, None)
        dist = F.log_softmax(outp, dim=1)
        _, argmax = dist.max(dim=1)

        all_true.extend(target.cpu().numpy())
        all_pred.extend(argmax.cpu().numpy())

        # Track conversation lengths
        s_masks_tensor = torch.stack(s_masks)  # Shape: (T, B)
        s_masks_np = s_masks_tensor.detach().cpu().numpy()

        for i in range(s_masks_np.shape[1]):
            active_steps = int(np.sum(s_masks_np[:, i] > 0))
            label = target[i].item()
            conv_lengths[label].append(active_steps)

    # 🎯 Compute per-class F1 scores
    f1_scores = f1_score(all_true, all_pred, average=None)  # Returns array of per-class F1

    # 🎯 Compute average conversation lengths
    mean_conv_lengths = convmean_per_class(conv_lengths)  # {class_id: mean_length}

    # Prepare data for plotting
    classes = sorted(mean_conv_lengths.keys())
    conv_means = [mean_conv_lengths[c] for c in classes]
    f1_vals = [f1_scores[c] for c in classes]

    # Plot F1 vs Conversation Length
    plt.figure(figsize=(8, 6))
    plt.scatter(conv_means, f1_vals)

    for i, cls in enumerate(classes):
        plt.annotate(f'Class {cls}', (conv_means[i], f1_vals[i]))

    plt.xlabel('Average Conversation Length')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Conversation Length per Class')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('f1_vs_conversation_length.png')
    plt.close()
    print("Saved plot as f1_vs_conversation_length.png")

                
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
        print(f"Class {cid}\nMean: {mean_len:.2f}, Variance: {var_len:.2f}\n")

    return mean_per_class
        
def get_rec_outp(y, masks):
    def negent(yy):
        probs = F.softmax(yy)
        return (torch.log(probs + 1e-8) * probs).sum(1).mean()

    negentropy = map(negent, y)

    if masks is not None:

        batch_size = y[0].size(0)
        exchange_steps = len(masks)

        inp = torch.cat([yy.view(batch_size, 1, -1) for yy in y], 1)
        mask = torch.cat(masks, 1).view(
            batch_size, exchange_steps, 1).expand_as(inp)
        outp = torch.masked_select(inp, mask.detach().bool()).view(batch_size, -1)
        return outp, negentropy
    else:
        return y[-1], negentropy


def entropy_from_log_probs(log_probs):
    probs = log_probs.exp()
    entropy = -(log_probs * probs).sum(dim=1)
    return [e.unsqueeze(0) for e in entropy]



import torch
import torch.nn.functional as F

def compute_similarities(messages):
    n = len(messages)
    hamming_sims = []
    cosine_sims = []
    jaccard_sims = []

    # Flatten messages to (50,) tensors
    messages = [msg[0].flatten() for msg in messages]

    for i in range(n):
        for j in range(i + 1, n):
            x = messages[i]
            y = messages[j]

            # Hamming Similarity
            hamming_distance = (x != y).sum().item()
            hamming_sim = (50 - hamming_distance) / 50
            hamming_sims.append(hamming_sim)

            # Cosine Similarity
            cos_sim = F.cosine_similarity(x.float(), y.float(), dim=0).item()
            cosine_sims.append(cos_sim)

            # Jaccard Similarity
            intersection = torch.logical_and(x, y).sum().item()
            union = torch.logical_or(x, y).sum().item()
            jaccard_sim = intersection / union if union != 0 else 1.0
            jaccard_sims.append(jaccard_sim)

    print(f"Average Hamming Similarity: {sum(hamming_sims) / len(hamming_sims):.4f}")
    print(f"Average Cosine Similarity:  {sum(cosine_sims) / len(cosine_sims):.4f}")
    print(f"Average Jaccard Similarity: {sum(jaccard_sims) / len(jaccard_sims):.4f}")

    return sum(hamming_sims) / len(hamming_sims), sum(cosine_sims) / len(cosine_sims),sum(jaccard_sims) / len(jaccard_sims)

def print_first_message(classno=0):
    print("Calculating distances for class "+str(classno))
    first_messages = []
    for idx, batch in enumerate(test_loader):
        audio, distractors, target, correct_index = batch

        if target.item() == classno:
            exchange_args = {
                "audio": audio,  
                "target": target,
                "distractors": distractors,
                "desc": None,  
                "train": False,
                "break_early": False
            }

            s, sen_w, rec_w, y, bs, br = conversation(sender, receiver, baseline_sen, baseline_rec, exchange_args)
            sen_feats, sen_probs = sen_w
            rec_feats, rec_probbs = rec_w

            first_messages.append(sen_feats[0])

    if len(first_messages) < 2:
        print("Not enough messages to compute similarities.")
        return

    ham, thing, thing2 = compute_similarities(first_messages)
    
    return ham, thing, thing2


def print_inter_class_similarity(class1=0, class2=1):
    print(f"Calculating inter-class similarities between class {class1} and class {class2}...")

    class1_messages = []
    class2_messages = []

    for idx, batch in enumerate(test_loader):
        audio, distractors, target, correct_index = batch
        class_label = target.item()

        if class_label == class1 or class_label == class2:
            exchange_args = {
                "audio": audio,
                "target": target,
                "distractors": distractors,
                "desc": None,
                "train": False,
                "break_early": False
            }

            s, sen_w, rec_w, y, bs, br = conversation(sender, receiver, baseline_sen, baseline_rec, exchange_args)
            sen_feats, sen_probs = sen_w

            if class_label == class1:
                class1_messages.append(sen_feats)
            else:
                class2_messages.append(sen_feats)

    if not class1_messages or not class2_messages:
        print("Not enough messages from both classes to compute similarities.")
        return

    # Flatten messages to (50,) tensors
    class1_messages = [msg[0].flatten() for msg in class1_messages]
    class2_messages = [msg[0].flatten() for msg in class2_messages]

    hamming_sims = []
    cosine_sims = []
    jaccard_sims = []

    for msg1 in class1_messages:
        for msg2 in class2_messages:
            # Hamming
            hamming_distance = (msg1 != msg2).sum().item()
            hamming_sim = (50 - hamming_distance) / 50
            hamming_sims.append(hamming_sim)

            # Cosine
            cos_sim = F.cosine_similarity(msg1.float(), msg2.float(), dim=0).item()
            cosine_sims.append(cos_sim)

            # Jaccard
            intersection = torch.logical_and(msg1, msg2).sum().item()
            union = torch.logical_or(msg1, msg2).sum().item()
            jaccard_sim = intersection / union if union != 0 else 1.0
            jaccard_sims.append(jaccard_sim)

    print(f"\nAverage inter-class similarity ({class1} vs {class2}):")
    print(f"  Hamming: {sum(hamming_sims) / len(hamming_sims):.4f}")
    print(f"  Cosine:  {sum(cosine_sims) / len(cosine_sims):.4f}")
    print(f"  Jaccard: {sum(jaccard_sims) / len(jaccard_sims):.4f}")

    return sum(hamming_sims) / len(hamming_sims), sum(cosine_sims) / len(cosine_sims), sum(jaccard_sims) / len(jaccard_sims)

import matplotlib.pyplot as plt
import numpy as np

def generate_and_save_hamming_matrix(num_classes, message_size=50):
    hamming_matrix = np.zeros((num_classes, num_classes))
    cosine_matrix = np.zeros((num_classes, num_classes))
    jaccard_matrix = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                print(f"Computing intra-class similarities for class {i}")
                hamming_sim, cosine, jaccard = print_first_message(i)
            else:
                print(f"Computing inter-class similarities between class {i} and class {j}")
                hamming_sim, cosine, jaccard = print_inter_class_similarity(i, j)

            hamming_matrix[i, j] = hamming_sim
            cosine_matrix[i, j] = cosine
            jaccard_matrix[i, j] = jaccard

    def plot_and_save(matrix, title, filename, colorbar_label):
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label=colorbar_label)
        plt.title(f'{title} (Message Size = {message_size})')
        plt.xlabel('Class')
        plt.ylabel('Class')
        plt.xticks(ticks=np.arange(num_classes), labels=np.arange(num_classes))
        plt.yticks(ticks=np.arange(num_classes), labels=np.arange(num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center',
                         color='white' if matrix[i, j] < 0.5 else 'black')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"{title} matrix saved as {filename}")

    plot_and_save(hamming_matrix, "Hamming Similarity Matrix", f"message-size-{message_size}-hamming.png", "Average Hamming Similarity")
    plot_and_save(cosine_matrix, "Cosine Similarity Matrix", f"message-size-{message_size}-cosine.png", "Average Cosine Similarity")
    plot_and_save(jaccard_matrix, "Jaccard Similarity Matrix", f"message-size-{message_size}-jaccard.png", "Average Jaccard Similarity")


def track_bit_usage(classno=0):
    print(f"Tracking bit usage for class {classno}...")
    collected_messages = []

    for idx, batch in enumerate(test_loader):
        audio, distractors, target, correct_index = batch

        if target.item() == classno:
            exchange_args = {
                "audio": audio,  
                "target": target,
                "distractors": distractors,
                "desc": None,  
                "train": False,
                "break_early": False
            }

            s, sen_w, rec_w, y, bs, br = conversation(sender, receiver, baseline_sen, baseline_rec, exchange_args)
            sen_feats, sen_probs = sen_w  # Assuming sen_feats is the binary message (tensor)

            collected_messages.append(sen_feats[0].cpu().detach())


    if len(collected_messages) == 0:
        print("No messages collected for this class.")
        return

    # Stack messages into a tensor [num_samples, message_length]
    messages_tensor = torch.stack(collected_messages).squeeze(1)

    # Compute bit usage frequency
    bit_usage = messages_tensor.float().mean(dim=0)  # Mean activation per bit

    print(f"Bit usage for class {classno}:")
    print(bit_usage)

    return bit_usage


def plot_bit_usage(bit_usage, classno):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(bit_usage)), bit_usage.numpy())
    plt.ylim(0, 1)
    plt.title(f"Bit Usage Frequency for Class {classno}")
    plt.xlabel("Bit Position")
    plt.ylabel("Activation Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import numpy as np

def compute_mean_bit_entropy(bit_usages):
    eps = 1e-8
    if isinstance(bit_usages, torch.Tensor):
        bit_usages = bit_usages.cpu().numpy()

    entropies = - (bit_usages * np.log2(bit_usages + eps) + (1 - bit_usages) * np.log2(1 - bit_usages + eps))
    return np.mean(entropies)

def count_zeros_ones(tensor_vals):
    if isinstance(tensor_vals, torch.Tensor):
        tensor_vals = tensor_vals.cpu()

    num_ones = (tensor_vals == 1.0).sum().item()
    num_zeros = (tensor_vals == 0.0).sum().item()

    print(f"Number of 1's  : {num_ones}")
    print(f"Number of 0's  : {num_zeros}")

import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def compute_pca_redundancy(messages, variance_threshold=0.95):
    """
    Perform PCA on sender messages to estimate redundancy.
    
    Args:
        messages (Tensor): shape [num_samples, message_size]
        variance_threshold (float): e.g., 0.95 for 95% variance explained.
    
    Returns:
        effective_dim (int): Number of components explaining the threshold variance.
        explained_variance (ndarray): Cumulative variance explained.
    """
    messages_np = messages.cpu().numpy()

    pca = PCA()
    pca.fit(messages_np)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    effective_dim = np.searchsorted(cumulative_variance, variance_threshold) + 1

    # Plot variance explained
    plt.figure(figsize=(8,5))
    plt.plot(cumulative_variance, marker='o')
    plt.axhline(y=variance_threshold, color='r', linestyle='--')
    plt.axvline(x=effective_dim, color='g', linestyle='--')
    plt.title("PCA Cumulative Explained Variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.grid(True)
    plt.show()

    print(f"Effective Dimensionality (>{variance_threshold*100}% variance): {effective_dim}")
    return effective_dim, cumulative_variance

def collect_first_messages(classno=0):
    print(f"Collecting first messages for class {classno}...")
    collected_messages = []

    for idx, batch in enumerate(test_loader):
        audio, distractors, target, correct_index = batch

        if target.item() == classno:
            exchange_args = {
                "audio": audio,  
                "target": target,
                "distractors": distractors,
                "desc": None,  
                "train": False,
                "break_early": False
            }

            s, sen_w, rec_w, y, bs, br = conversation(sender, receiver, baseline_sen, baseline_rec, exchange_args)
            sen_feats, sen_probs = sen_w  # sen_feats: list of sender messages over time

            # Collect ONLY the first message
            first_message = sen_feats[0].cpu().detach()  # Shape: [1, message_length] or [message_length]
            collected_messages.append(first_message.squeeze(0))

    if len(collected_messages) == 0:
        print("No messages collected for this class.")
        return None

    # Stack messages into tensor [num_samples, message_length]
    messages_tensor = torch.stack(collected_messages)

    print(f"Collected {messages_tensor.size(0)} first messages for class {classno}.")
    return messages_tensor


def print_convmeanlengths():
    """
    Evaluates sender and receiver models and collects conversation lengths per class label.
    Returns a dictionary {class_label: [lengths]}.
    """
    fixed_exchange = False
    sender.eval()
    receiver.eval()

    conv_lengths = defaultdict(list)

    with torch.no_grad():
        for batch in test_loader:
            audio, distractors, target, correct_index = batch

            exchange_args = {
                "audio": audio,
                "target": target,
                "distractors": distractors,
                "desc": None,
                "train": False,
                "break_early": False
            }

            s, sen_w, rec_w, y, bs, br = conversation(sender, receiver, baseline_sen, baseline_rec, exchange_args)
            s_masks, _, _ = s

            # Count active steps per sample in batch
            s_masks_tensor = torch.stack(s_masks)  # Shape: (T, B)
            s_masks_np = s_masks_tensor.detach().cpu().numpy()
    
            for i in range(s_masks_np.shape[1]):
                active_steps = int(np.sum(s_masks_np[:, i] > 0))
                label = target[i].item()
                conv_lengths[label].append(active_steps)
                
    track_conversation_length(conv_lengths)
    return conv_lengths

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
        print(f"Class {cid}\nMean: {mean_len}, Variance: {var_len:.2f}\n")


# mean_similarity = 0
# for i in range(0,6):
#     ham, cosine, jaccard =  print_first_message(i)
#     mean_similarity = mean_similarity+cosine
# print(mean_similarity/6)
# print_convmeanlengths()



#entropy_per_conv()
f1_vs_convlength()
#print_first_message(0)
#print_inter_class_similarity(0, 1)
generate_and_save_hamming_matrix(num_classes=6, message_size=message_size)

#bit_usage = track_bit_usage(classno=0)
#plot_bit_usage(bit_usage, classno=0)
#print(compute_mean_bit_entropy(bit_usage))
#count_zeros_ones(bit_usage)

#compute_pca_redundancy(collect_first_messages(classno=5))
















