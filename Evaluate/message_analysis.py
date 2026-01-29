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
runfile = 'msg_size_10_uni/'

#-------------Game Settings--------------#
unimodal = False
audio = True # if audio is sender
dynamic = True
sender_is_learning = True # Funny that this actually came from a bug

n_classes = 6
n_distractors = 2
message_size = 10
max_conv_length=1
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


img_emb = 'vgg_image_embeddings-zm.npz'
aud_emb = 'vggish_dual_labels_embeddings-pca.npz'


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



class SyntheticData2(Dataset):
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
        self.labels = data['shape_labels']
        self.freq_labels = data['freq_labels']

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
        freq_label = int(self.freq_labels[idx])

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
        
        return (
        target_embedding.to(device),
        dist_embeddings.to(device),
        torch.tensor(label, dtype=torch.long).to(device),
        torch.tensor(correct_image_index, dtype=torch.long).to(device),
        torch.tensor(freq_label, dtype=torch.long).to(device)  # ✅ Include frequency label
    )






#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

dataset = SyntheticData2(n_distractors,audio_embedding_file=aud_emb,image_embedding_file=img_emb)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)



#------------------------------------------Conversation Function------------------------------------------#
def conversation(sender, receiver, baseline_sen, baseline_rec, exchange_args: Dict[str, Any],bit_activity=None,num_bits_to_flip=None,typeof=None,number=None):
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

        if bit_activity is not None and i_exch == 0:
            if typeof == 'active':
                condition = (bit_activity <= 0.1) | (bit_activity >= 0.9)
                if number==0:
                    condition = (bit_activity <= 0.1) 
                elif number==1:
                    condition = (bit_activity >= 0.9)
                indices = torch.nonzero(condition).flatten()
            
                if len(indices) < num_bits_to_flip:
                    selected_indices = indices.tolist() 
                else:
                    selected_indices = random.sample(indices.tolist(), num_bits_to_flip)
                z_binary[0, selected_indices] = 1 - z_binary[0, selected_indices]
            
            elif typeof == 'variable':
                condition = (bit_activity < 0.9) & (bit_activity > 0.1)
                indices = torch.nonzero(condition).flatten()
            
                if len(indices) < num_bits_to_flip:
                    pass  # Not enough bits to flip
                else:
                    selected_indices = random.sample(indices.tolist(), num_bits_to_flip)
                    z_binary[0, selected_indices] = 1 - z_binary[0, selected_indices]


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



#-----------------------------------------------------------------------------------------------------------------------------------------#


def run_perturbation_trials(num_bits_to_flip=0, typeof='active', number=2, num_trials=10):
    active_accuracy = []      # List of overall accuracies
    active_class_accuracy = []  # List of per-class accuracy lists

    for i in range(num_trials):
        acc, classacc = eval_system(num_bits_to_flip, typeof,number)   # Pass number of bits to flip
        active_accuracy.append(acc)
        active_class_accuracy.append(classacc)

    # Convert to numpy arrays
    active_accuracy = np.array(active_accuracy)                      # Shape: [num_trials]
    active_class_accuracy = np.array(active_class_accuracy)          # Shape: [num_trials, num_classes]

    # Compute mean and variances
    active_mean = active_accuracy.mean()
    active_variance = active_accuracy.var()

    active_mean_per_class = active_class_accuracy.mean(axis=0)
    active_variance_per_class = active_class_accuracy.var(axis=0)

    # === Print results ===
    mean_list = [round(float(val), 2) for val in active_mean_per_class]
    var_list = [round(float(val), 2) for val in active_variance_per_class]
    
    print(f"\nResults for flipping {num_bits_to_flip} bits, type = '{typeof}'")
    print(f"Overall Accuracy Mean: {active_mean:.2f}")
    print(f"Overall Accuracy Variance: {active_variance:.2f}")
    print(f"Per-Class Accuracy Mean: {mean_list}")
    print(f"Per-Class Accuracy Variance: {var_list}")
    
    return active_mean, active_variance, active_mean_per_class, active_variance_per_class



def eval_system(num_bits_to_flip, typeof,number):
    correct_pred = 0 
    total_samples = 0
    class_accuracies = []

    for i in range(6):   # For each class (0 to 5)
        bit_usage = track_bit_usage(i)
        cp, ts = eval_class(i, bit_usage, num_bits_to_flip, typeof,number)
        correct_pred += cp
        total_samples += ts
        class_accuracies.append((cp / ts) * 100 if ts > 0 else 0)

    accuracy = (correct_pred / total_samples) * 100 if total_samples > 0 else 0
    return accuracy, class_accuracies


def eval_class(classno, bit_activity, num_bits_to_flip, typeof,number):
    correct_predictions = 0
    total_samples = 0
    total_loss = 0
    
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

            s, sen_w, rec_w, y, bs, br = conversation(
                sender, receiver, baseline_sen, baseline_rec, 
                exchange_args, bit_activity, num_bits_to_flip, typeof,number)

            s_masks, s_feats, s_probs = s
            sen_feats, sen_probs = sen_w  

            if fixed_exchange:
                binary_s_masks = binary_rec_masks = binary_sen_masks = None
                bas_rec_masks = bas_sen_masks = None
                y_masks = None
            else:
                binary_s_masks = s_masks[:-1]
                binary_rec_masks = s_masks[1:-1]
                binary_sen_masks = s_masks[:-1]
                bas_rec_masks = s_masks[:-1]
                bas_sen_masks = s_masks[:-1]
                y_masks = [torch.min(1 - m1, m2) for m1, m2 in zip(s_masks[1:], s_masks[:-1])]

            outp, ent_y_rec = get_rec_outp(y, y_masks)

            # Compute loss
            dist = F.log_softmax(outp, dim=1)
            nll_loss = nn.NLLLoss()(dist, target)

            # Compute accuracy
            maxdist, argmax = dist.max(dim=1)
            correct_predictions += (argmax == target).sum().item()
            total_samples += target.size(0)
            total_loss += nll_loss.item()

    return correct_predictions, total_samples


def track_bit_usage(classno=0):
    #print(f"Tracking bit usage for class {classno}...")
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

    #print(f"Bit usage for class {classno}:")
    #print(bit_usage)

    return bit_usage

def track_bit_usage_interclass(classno=0,freq=0):
    #print(f"Tracking bit usage for class {classno}...")
    collected_messages = []

    for idx, batch in enumerate(test_loader):
        audio, distractors, target, correct_index, freq_index = batch

        if target.item() == classno:
            exchange_args = {
                "audio": audio,  
                "target": target,
                "distractors": distractors,
                "desc": None,  
                "train": False,
                "break_early": False
            }

            #if freq_index.item() not in [1, 2]:
            if freq is not None and freq_index.item() != freq:
                continue

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

    #print(f"Bit usage for class {classno}:")
    #print(bit_usage)

    return bit_usage




import numpy as np
import matplotlib.pyplot as plt

def perturbation_analysis_plot(typeof='active', num_trials=10, save_path_prefix='perturbation_plot'):
    max_bits = 10   # Assuming message length is 10 bits
    bit_flip_range = range(0, max_bits + 1)   # Flip from 0 to 10 bits
    class_labels = ['Circle', 'Heart', 'Hexagon', 'Square', 'Star', 'Triangle']

    numbers = [0, 1]   # 0s and 1s
    linestyles = {0: 'dotted', 1: 'solid'}

    colors = plt.get_cmap('tab10').colors   # Get 6 distinct colors

    # === Plot 1: Per-Class Accuracy ===
    plt.figure(figsize=(10, 6))

    for number in numbers:
        all_means_per_class = []
        overall_variances = []

        for num_bits in bit_flip_range:
            print(f"\nRunning trials for flipping bits in {number}s, flipping {num_bits} bits...")
            _, variance, mean_per_class, _ = run_perturbation_trials(num_bits_to_flip=num_bits, typeof=typeof, number=number, num_trials=num_trials)
            all_means_per_class.append(mean_per_class)
            overall_variances.append(variance)

        all_means_per_class = np.array(all_means_per_class)

        # Plot each class with same color, different linestyle
        for cls in range(len(class_labels)):
            plt.plot(bit_flip_range, all_means_per_class[:, cls],
                     label=f'{class_labels[cls]} ({number}s)',
                     linestyle=linestyles[number],
                     color=colors[cls])

    plt.title(f'Per-Class Accuracy vs Number of Bits Flipped ({typeof})')
    plt.xlabel('Number of Bits Flipped')
    plt.ylabel('Per-Class Mean Accuracy (%)')
    plt.ylim(0, 100)
    plt.legend(title="Class (Bits Flipped)")
    plt.grid(True)
    plt.tight_layout()
    acc_path = f"{save_path_prefix}_accuracy_comparison.png"
    plt.savefig(acc_path)
    print(f"\nPer-class accuracy comparison plot saved to {acc_path}")
    plt.show()

    # === Plot 2: Overall Variance ===
    plt.figure(figsize=(10, 6))

    for number in numbers:
        overall_variances = []

        for num_bits in bit_flip_range:
            _, variance, _, _ = run_perturbation_trials(num_bits_to_flip=num_bits, typeof=typeof, number=number, num_trials=num_trials)
            overall_variances.append(variance)

        overall_variances = np.array(overall_variances)

        plt.plot(bit_flip_range, overall_variances,
                 label=f'Flipping {number}s',
                 linestyle=linestyles[number],
                 color='black')

    plt.title(f'Overall Accuracy Variance vs Number of Bits Flipped ({typeof})')
    plt.xlabel('Number of Bits Flipped')
    plt.ylabel('Accuracy Variance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    var_path = f"{save_path_prefix}_variance_comparison.png"
    plt.savefig(var_path)
    print(f"Overall variance comparison plot saved to {var_path}")
    plt.show()







#----------------------- TSNE PLOTS ----
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def collect_and_plot_message_space():
    print("Collecting sender messages for all classes...")

    collected_messages = []
    collected_labels = []

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
        sen_feats, sen_probs = sen_w  # Assuming sen_feats is [1, message_length]

        collected_messages.append(sen_feats[0].cpu().detach())
        collected_labels.append(target.item())

    if len(collected_messages) == 0:
        print("No messages collected.")
        return

    # Stack messages into tensor
    messages_tensor = torch.stack(collected_messages).squeeze(1)   # Shape: [N, message_length]
    labels_array = np.array(collected_labels)

    # === Apply t-SNE ===
    tsne = TSNE(n_components=2, random_state=42)
    messages_2d = tsne.fit_transform(messages_tensor.numpy())

    # === Plot ===
    class_labels = ['Circle', 'Heart', 'Hexagon', 'Square', 'Star', 'Triangle']
    unique_classes = np.unique(labels_array)

    plt.figure(figsize=(10, 8))
    for cls in unique_classes:
        idx = labels_array == cls
        plt.scatter(messages_2d[idx, 0], messages_2d[idx, 1], label=class_labels[cls], alpha=0.7)

    plt.title('t-SNE of Unimodal Sender Messages by Class')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sender_messages_tsne.png')
    print("t-SNE plot saved as 'sender_messages_tsne_unimodal.png'")
    plt.show()


import torch
import numpy as np
import matplotlib.pyplot as plt
import umap

def collect_and_plot_message_umap(n_neighbors=15, min_dist=0.1, save_path='sender_messages_umap.png'):
    print("Collecting sender messages for UMAP visualization...")

    collected_messages = []
    collected_labels = []

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
        sen_feats, sen_probs = sen_w  # Assuming sen_feats is [1, message_length]

        collected_messages.append(sen_feats[0].cpu().detach())
        collected_labels.append(target.item())

    if len(collected_messages) == 0:
        print("No messages collected.")
        return

    # Stack messages into tensor
    messages_tensor = torch.stack(collected_messages).squeeze(1)   # Shape: [N, message_length]
    labels_array = np.array(collected_labels)

    # === Apply UMAP ===
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    messages_2d = reducer.fit_transform(messages_tensor.numpy())

    # === Plot ===
    class_labels = ['Circle', 'Heart', 'Hexagon', 'Square', 'Star', 'Triangle']
    unique_classes = np.unique(labels_array)

    plt.figure(figsize=(10, 8))
    for cls in unique_classes:
        idx = labels_array == cls
        plt.scatter(messages_2d[idx, 0], messages_2d[idx, 1], label=class_labels[cls], alpha=0.7)

    plt.title('UMAP of Sender Messages by Class')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"UMAP plot saved as '{save_path}'")
    plt.show()

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def collect_and_plot_message_pca(save_path='sender_messages_pca.png'):
    print("Collecting sender messages for PCA visualization...")

    collected_messages = []
    collected_labels = []

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
        sen_feats, sen_probs = sen_w  # Assuming sen_feats is [1, message_length]

        collected_messages.append(sen_feats[0].cpu().detach())
        collected_labels.append(target.item())

    if len(collected_messages) == 0:
        print("No messages collected.")





def collect_and_plot_message_space(classno):
    print("Collecting sender messages for 'Circle' class, grouped by frequency classes...")

    collected_messages = []
    collected_freq_labels = []

    for idx, batch in enumerate(test_loader):
        audio, distractors, target, correct_index, frequency = batch

        # Only process samples where target == 0 (Circle class)
        if target.item() != classno:
            continue

        exchange_args = {
            "audio": audio,  
            "target": target,
            "distractors": distractors,
            "desc": None,  
            "train": False,
            "break_early": False
        }

        s, sen_w, rec_w, y, bs, br = conversation(sender, receiver, baseline_sen, baseline_rec, exchange_args)
        sen_feats, sen_probs = sen_w  # Assuming sen_feats is [1, message_length]

        collected_messages.append(sen_feats[0].cpu().detach())
        collected_freq_labels.append(frequency.item())

    if len(collected_messages) == 0:
        print("No messages collected for Circle class.")
        return

    # Stack messages into tensor
    messages_tensor = torch.stack(collected_messages).squeeze(1)   # Shape: [N, message_length]
    freq_labels_array = np.array(collected_freq_labels)

    # === Apply t-SNE ===
    tsne = TSNE(n_components=2, random_state=42)
    messages_2d = tsne.fit_transform(messages_tensor.numpy())

    # === Plot by Frequency Class ===
    freq_class_labels = ['0.3 - 0.4 Amplitude', '0.5 - 0.6 Amplitude', '0.7 - 0.9 Amplitude']
    unique_freqs = np.unique(freq_labels_array)

    plt.figure(figsize=(10, 8))
    for freq_cls in unique_freqs:
        idx = freq_labels_array == freq_cls
        plt.scatter(messages_2d[idx, 0], messages_2d[idx, 1], label=freq_class_labels[freq_cls], alpha=0.7)

    plt.title(f't-SNE of Sender Messages for class {classno} by Amplitude')
    plt.legend(title="Frequency Class")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sender_messages_tsne_circle_freq.png')
    print("t-SNE plot saved as 'sender_messages_tsne_circle_freq.png'")
    plt.show()


from umap import UMAP  # Make sure you have umap-learn installed

def collect_and_plot_message_space_umap(classno):
    print("Collecting sender messages for 'Circle' class, grouped by frequency classes...")

    collected_messages = []
    collected_freq_labels = []

    for idx, batch in enumerate(test_loader):
        audio, distractors, target, correct_index, frequency = batch

        # Only process samples where target == 1 (Circle class)
        if target.item() != classno:
            continue

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

        collected_messages.append(sen_feats[0].cpu().detach())
        collected_freq_labels.append(frequency.item())

    if len(collected_messages) == 0:
        print("No messages collected for Circle class.")
        return

    # Stack messages into tensor
    messages_tensor = torch.stack(collected_messages).squeeze(1)   # Shape: [N, message_length]
    freq_labels_array = np.array(collected_freq_labels)

    # === Apply UMAP ===
    umap = UMAP(n_components=2, random_state=42)
    messages_2d = umap.fit_transform(messages_tensor.numpy())

    # === Plot by Frequency Class ===
    freq_class_labels = ['200-300 Hz', '400-500 Hz', '600-800 Hz']
    unique_freqs = np.unique(freq_labels_array)

    plt.figure(figsize=(10, 8))
    for freq_cls in unique_freqs:
        idx = freq_labels_array == freq_cls
        plt.scatter(messages_2d[idx, 0], messages_2d[idx, 1], label=freq_class_labels[freq_cls], alpha=0.7)

    plt.title('UMAP of Sender Messages for Circle Class by Frequency')
    plt.legend(title="Frequency Class")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sender_messages_umap_circle_freq.png')
    print("UMAP plot saved as 'sender_messages_umap_circle_freq.png'")
    plt.show()



import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torch
import matplotlib.colors as mcolors
import colorsys

def adjust_color_shade(base_color, lightness):
    """
    Adjust color brightness using HLS.
    - lightness: value between 0 (black) and 1 (white)
    """
    rgb = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    new_rgb = colorsys.hls_to_rgb(h, lightness, s)
    return new_rgb
    

def collect_and_plot_message_space_all():
    print("Collecting sender messages for all classes, using color shades for frequency classes...")

    collected_messages = []
    collected_shape_labels = []
    collected_freq_labels = []

    for idx, batch in enumerate(test_loader):
        audio, distractors, target, correct_index, frequency = batch

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

        collected_messages.append(sen_feats[0].cpu().detach())
        collected_shape_labels.append(target.item())
        collected_freq_labels.append(frequency.item())

    if len(collected_messages) == 0:
        print("No messages collected.")
        return

    messages_tensor = torch.stack(collected_messages).squeeze(1)
    shape_labels_array = np.array(collected_shape_labels)
    freq_labels_array = np.array(collected_freq_labels)

    # === Apply t-SNE ===
    tsne = TSNE(n_components=2, random_state=42)
    messages_2d = tsne.fit_transform(messages_tensor.numpy())

    # === Plot Setup ===
    shape_class_labels = ['Circle', 'Heart', 'Hexagon', 'Square', 'Star', 'Triangle']
    amp_class_labels = ['0.3 - 0.4 Amplitude', '0.5 - 0.6 Amplitude', '0.7 - 0.9 Amplitude']


    base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    shade_lightness = {
        0: 0.85,   # Very light
        1: 0.5,    # Normal
        2: 0.2     # Very dark
    }

    plt.figure(figsize=(12, 10))

    for shape_cls in np.unique(shape_labels_array):
        for freq_cls in np.unique(freq_labels_array):
            idx = (shape_labels_array == shape_cls) & (freq_labels_array == freq_cls)
            if not np.any(idx):
                continue
            base_color = base_colors[shape_cls-1]
            lightness = shade_lightness[freq_cls]
            color_shade = adjust_color_shade(base_color, lightness)

            plt.scatter(
                messages_2d[idx, 0],
                messages_2d[idx, 1],
                color=[color_shade],
                label=f"{shape_class_labels[shape_cls-1]} - {amp_class_labels[freq_cls]}",
                alpha=0.9
            )

    plt.title('t-SNE of Sender Messages: Shape Classes (Color) & Amplitude (Shade)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sender_messages_tsne_all_classes_shades.png')
    print("t-SNE plot saved as 'sender_messages_tsne_all_classes_shades.png'")
    plt.show()


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
    
#collect_and_plot_message_space_all()


plot_bit_usage(track_bit_usage_interclass(1,0),1)

#collect_and_plot_message_space(3)


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
        audio, distractors, target, correct_index, freq = batch

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

print_first_message(classno=2)
#eval_system(0.2,'active')
#run_perturbation_trials(0.3,'active')
#perturbation_analysis_plot(typeof='active', num_trials=10, save_path_prefix='active_perturbation')
#print(track_bit_usage(5))
#collect_and_plot_message_space
#collect_and_plot_message_umap(n_neighbors=10, min_dist=0.2, save_path='messages_umap_plot_multimodal.png')
#collect_and_plot_message_pca(save_path='sender_messages_pca_multimodal.png')

#collect_and_plot_message_space(0)
#collect_and_plot_message_space_umap(0)
