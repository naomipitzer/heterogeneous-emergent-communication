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

#-------------Game Settings--------------#
unimodal = True
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


#-------------Training Parameters-----------#
entropy_term = 0.1
entropy_stop = 0.1
num_epochs = 110
batch_size = 32
learning_rate = 1e-4
iterations_per_epoch = 0
optim_type = "RMSprop"
top_k_train = 3

print(f"Training parameters:\nEpochs={num_epochs}   BatchSize={batch_size}   LR={learning_rate}   ItPerEpoch:{iterations_per_epoch}\n")



# ------------- Dataset ------------#
audio_folder = 'data/synthetic_audio'
image_folder = 'data/synthetic_shapes'

# No pre-processing
#aud_emb = 'vggish_embeddings.npz'
#img_emb = 'vgg_image_embeddings.npz'

# Zero-meaning
#aud_emb = 'vggish_embeddings-zm.npz'
img_emb = 'cifar100_selected_embeddings-zm.npz'

# PCA
aud_emb = 'environmental-audio-pca.npz'

# Standardisation
#aud_emb = 'vggish_embeddings-st.npz'
#img_emb = 'vgg_image_embeddings-st.npz'

print(f"Audio Dataset:{audio_folder}     Image Dataset:{image_folder}\n")



#----------Embeddings------------#
print("Embeding generation models not needed anymore...")




#####------------ FIXED DISTRACTOR SET ------------#####
data = np.load(aud_emb)
embeddings = data['embeddings']
labels = data['labels']

imdata = np.load(img_emb)
imembeddings = imdata['embeddings']
imlabels = imdata['labels']

if not dynamic:
    if not audio:
        if unimodal:
            circle = torch.tensor(imembeddings[np.where(imlabels == 0)[0][0]], dtype=torch.float32)
            heart = torch.tensor(imembeddings[np.where(imlabels == 1)[0][0]], dtype=torch.float32)
            hexagon = torch.tensor(imembeddings[np.where(imlabels == 2)[0][0]], dtype=torch.float32)
            square = torch.tensor(imembeddings[np.where(imlabels == 3)[0][0]], dtype=torch.float32)
            star = torch.tensor(imembeddings[np.where(imlabels == 4)[0][0]], dtype=torch.float32)
            triangle = torch.tensor(imembeddings[np.where(imlabels == 5)[0][0]], dtype=torch.float32)
    
    
            emb_set = torch.stack([circle, heart, hexagon, square, star, triangle], dim=0).squeeze(1)
            print("Images shape: "+str(emb_set.shape))
        else: 
            circle = torch.tensor(embeddings[np.where(labels == 0)[0][0]], dtype=torch.float32)
            heart = torch.tensor(embeddings[np.where(labels == 1)[0][0]], dtype=torch.float32)
            hexagon = torch.tensor(embeddings[np.where(labels == 2)[0][0]], dtype=torch.float32)
            square = torch.tensor(embeddings[np.where(labels == 3)[0][0]], dtype=torch.float32)
            star = torch.tensor(embeddings[np.where(labels == 4)[0][0]], dtype=torch.float32)
            triangle = torch.tensor(embeddings[np.where(labels == 5)[0][0]], dtype=torch.float32)
    
    
            emb_set = torch.stack([circle, heart, hexagon, square, star, triangle], dim=0).squeeze(1)
            print("Audio shape: "+str(emb_set.shape))
        
    else: 
        if unimodal:
            circle = torch.tensor(embeddings[np.where(labels == 0)[0][0]], dtype=torch.float32)
            heart = torch.tensor(embeddings[np.where(labels == 1)[0][0]], dtype=torch.float32)
            hexagon = torch.tensor(embeddings[np.where(labels == 2)[0][0]], dtype=torch.float32)
            square = torch.tensor(embeddings[np.where(labels == 3)[0][0]], dtype=torch.float32)
            star = torch.tensor(embeddings[np.where(labels == 4)[0][0]], dtype=torch.float32)
            triangle = torch.tensor(embeddings[np.where(labels == 5)[0][0]], dtype=torch.float32)
    
    
            emb_set = torch.stack([circle, heart, hexagon, square, star, triangle], dim=0).squeeze(1)
            print("Audio shape: "+str(emb_set.shape))
        if not unimodal:
            circle = torch.tensor(imembeddings[np.where(imlabels == 0)[0][0]], dtype=torch.float32)
            heart = torch.tensor(imembeddings[np.where(imlabels == 1)[0][0]], dtype=torch.float32)
            hexagon = torch.tensor(imembeddings[np.where(imlabels == 2)[0][0]], dtype=torch.float32)
            square = torch.tensor(imembeddings[np.where(imlabels == 3)[0][0]], dtype=torch.float32)
            star = torch.tensor(imembeddings[np.where(imlabels == 4)[0][0]], dtype=torch.float32)
            triangle = torch.tensor(imembeddings[np.where(imlabels == 5)[0][0]], dtype=torch.float32)
    
    
            emb_set = torch.stack([circle, heart, hexagon, square, star, triangle], dim=0).squeeze(1)
            print("Images shape: "+str(emb_set.shape))
else:
    emb_set=None
    print("Using dynamic distractor set")



#------------------------------------------Data Loader------------------------------------------#

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











#------------------------------------------Conversation Function------------------------------------------#
def conversation(sender, receiver, baseline_sen, baseline_rec, exchange_args: Dict[str, Any]):
    """
    Handles a communication exchange between a sender and a receiver.
    The sender encodes an audio feature into a message, which is interpreted by the receiver.
    
    Parameters:
        sender: Sender model (generates communication)
        receiver: Receiver model (interprets message from sender)
        baseline_sen: Baseline sender model
        baseline_rec: Baseline receiver model
        exchange_args: Dictionary containing all input arguments - from Evtimova's implementation

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




#------------------------------------------Training Loop------------------------------------------#
# Function to display resource usage
def print_resource_usage():
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"RAM Usage: {psutil.virtual_memory().percent}%")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")
    
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")


# Sorry...
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def train(sender, receiver, baseline_sen, baseline_rec, train_loader, test_loader,
          num_epochs=500, learning_rate=1e-4, batch_size=32, optim_type="RMSprop"):
    """
    Train sender and receiver using reinforcement learning loss and entropy regularization.
    """

    # Optimizer selection
    optimizers = {
        "SGD": lambda params: optim.SGD(params, lr=learning_rate),
        "Adam": lambda params: optim.Adam(params, lr=learning_rate),
        "RMSprop": lambda params: optim.RMSprop(params, lr=learning_rate)
    }
    
    if optim_type not in optimizers:
        raise ValueError("Unsupported optimizer type. Choose from 'SGD', 'Adam', or 'RMSprop'.")

    optimizer_rec = optimizers[optim_type](receiver.parameters())
    optimizer_sen = optimizers[optim_type](sender.parameters())
    optimizer_bas_rec = optimizers[optim_type](baseline_rec.parameters())
    optimizer_bas_sen = optimizers[optim_type](baseline_sen.parameters())

    Tmax = max_conv_length  # Max length of conversation

    print("Starting training loop...")
    torch.autograd.set_detect_anomaly(True)
    training_accuracies = []
    training_losses = []
    testing_accuracies = []
    testing_losses = []
    epoch_times = []
    receiver_entropies = []
    sender_entropies = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        batch_loader = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_no, batch in batch_loader:
            audio, distractors, target, correct_index = batch  

            exchange_args = {
                "audio": audio,  
                "target": target,
                "distractors": distractors,
                "desc": None,  
                "train": True,
                "break_early": False
            }

            # Forward pass through conversation
            s, sen_w, rec_w, y, bs, br = conversation(sender, receiver, baseline_sen, baseline_rec, exchange_args)

            s_masks, s_feats, s_probs = s
            
            
            sen_feats, sen_probs = sen_w
            rec_feats, rec_probs = rec_w

            # Mask loss if dynamic exchange length
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

            # Obtain predictions
            dist = F.log_softmax(outp, dim=1)
            maxdist, argmax = dist.max(dim=1)
            
            classification_entropy = entropy_from_log_probs(dist) # Calculate classification entropy
            message_entropy = entropy_from_message(sen_probs)


            # Receiver classification loss
            nll_loss = nn.NLLLoss()(dist, target)

            # Individual log-likelihoods across the batch
            logs = loglikelihood(dist, target).detach()

            if not fixed_exchange:
                loss_binary_s, ent_binary_s = multistep_loss_binary(s_feats, s_probs, logs, br, binary_s_masks, entropy_stop)
                
            if rec_feats[:-1]:  
                loss_binary_rec, ent_binary_rec = multistep_loss_binary(rec_feats[:-1], rec_probs[:-1], logs, br[:-1], 
                                                                        binary_rec_masks, entropy_term)
            else:
                loss_binary_rec, ent_binary_rec = (torch.zeros(1), []  )
                

            loss_binary_sen, ent_binary_sen = multistep_loss_binary(sen_feats, sen_probs, logs, bs, binary_sen_masks, 0.1)
            
            loss_bas_rec = multistep_loss_bas(br, logs, bas_rec_masks)
            loss_bas_sen = multistep_loss_bas(bs, logs, bas_sen_masks)

            loss_rec = nll_loss + loss_binary_rec
            if not fixed_exchange: 
                loss_rec = loss_rec + loss_binary_s

            
            # Update receiver
            optimizer_rec.zero_grad()
            loss_rec.backward()
            nn.utils.clip_grad_norm_(receiver.parameters(), max_norm=1.0)
            optimizer_rec.step()


            if sender_is_learning:
                loss_sen = loss_binary_sen
            else:
                loss_sen = loss_binary_sen.detach().clone().requires_grad_(True)
            loss_sen = loss_binary_sen
            # print(f"loss_sen grad_fn: {loss_sen.grad_fn}")
            
            # Update sender
            optimizer_sen.zero_grad()
            loss_sen.backward()
            nn.utils.clip_grad_norm_(sender.parameters(), max_norm=1.0)
            optimizer_sen.step()

            # Update baseline models
            optimizer_bas_rec.zero_grad()
            loss_bas_rec.backward()
            nn.utils.clip_grad_norm_(baseline_rec.parameters(), max_norm=1.0)
            optimizer_bas_rec.step()

            optimizer_bas_sen.zero_grad()
            loss_bas_sen.backward()
            nn.utils.clip_grad_norm_(baseline_sen.parameters(), max_norm=1.0)
            optimizer_bas_sen.step()

            # Compute top-k accuracy
            top_k_ind = torch.topk(dist, top_k_train, dim=1).indices
            target_exp = target.view(-1, 1).expand(-1, top_k_train)
            accuracy = (top_k_ind == target_exp).sum().item() / target.size(0)
            
            total_loss += loss_rec.item()
            correct_predictions += (argmax == target).sum().item()
            total_samples += target.size(0)

            batch_loader.set_postfix({
                "Loss": f"{total_loss / (batch_no + 1):.4f}",
                "Acc": f"{correct_predictions / total_samples * 100:.2f}%"
            })


        epoch_duration = time.time() - epoch_start_time
        epoch_accuracy = (correct_predictions / total_samples * 100) if total_samples > 0 else 0

        #Entropy
            # Receiver, Sender entropy
        if classification_entropy:
            avg_receiver_entropy = torch.cat(classification_entropy).mean().item()
        else:
            avg_receiver_entropy = 0.0

        if message_entropy:
            avg_sender_entropy = torch.cat(message_entropy).mean().item()
        else:
            avg_sender_entropy = 0.0

        receiver_entropies.append(avg_receiver_entropy)
        sender_entropies.append(avg_sender_entropy)

        training_accuracies.append(epoch_accuracy)
        training_losses.append(total_loss / len(train_loader))
        epoch_times.append(epoch_duration)
        
        print(f"Epoch {epoch + 1} finished in {epoch_duration:.2f} seconds.")
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {epoch_accuracy:.2f}%")
        print(f"Avg Receiver Entropy: {avg_receiver_entropy:.4f}, Avg Sender Entropy: {avg_sender_entropy:.4f}")

        # **Run testing**
        test_loss, test_accuracy, top3 = evaluate(sender, receiver, test_loader)
        testing_accuracies.append(test_accuracy)
        testing_losses.append(test_loss)

    # Plot accuracy and loss
    lengths_over_time(conv_lengths_epochs)
    plot_accuracy(training_accuracies, testing_accuracies, save_path='accuracy_plot.png')
    plot_loss(training_losses, testing_losses, save_path='loss_plot.png')
    plot_entropy(receiver_entropies, sender_entropies, save_path='entropy_plot.png')
    plot_epoch_times(epoch_times, save_path='epoch_times.png')

    total_duration = time.time() - start_time
    print(f"\nTraining completed in {total_duration / 60:.2f} minutes.")



def evaluate(sender, receiver, test_loader):
    """
    Evaluates the sender and receiver models on the test set.
    Now also prints Top-k Accuracy.
    """
    from collections import defaultdict
    import torch.nn.functional as F
    import torch.nn as nn

    fixed_exchange = False
    sender.eval()
    receiver.eval()

    total_loss = 0
    correct_predictions = 0
    correct_top3 = 0
    total_samples = 0
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

            s_masks, s_feats, s_probs = s
            sen_feats, sen_probs = sen_w
            rec_feats, rec_probs = rec_w

            if fixed_exchange:
                binary_s_masks = binary_rec_masks = binary_sen_masks = None
                bas_rec_masks = bas_sen_masks = None
                y_masks = None
            else:
                y_masks = [torch.min(1 - m1, m2) for m1, m2 in zip(s_masks[1:], s_masks[:-1])]

            outp, ent_y_rec = get_rec_outp(y, y_masks)

            # Compute loss
            dist = F.log_softmax(outp, dim=1)
            nll_loss = nn.NLLLoss()(dist, target)

            # Compute Top-1 accuracy
            maxdist, argmax = dist.max(dim=1)
            correct_predictions += (argmax == target).sum().item()

            # Compute Top-3 accuracy
            top3_vals, top3_indices = torch.topk(dist, k=top_k_train, dim=1)
            for i in range(target.size(0)):
                if target[i].item() in top3_indices[i].cpu().numpy():
                    correct_top3 += 1

            total_samples += target.size(0)
            total_loss += nll_loss.item()

            # Count active steps
            s_masks_tensor = torch.stack(s_masks)
            s_masks_np = s_masks_tensor.detach().cpu().numpy()
    
            for i in range(s_masks_np.shape[1]):
                active_steps = int(np.sum(s_masks_np[:, i] > 0))
                label = target[i].item()
                conv_lengths[label].append(active_steps)

    avg_loss = total_loss / len(test_loader)
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    top3_accuracy = (correct_top3 / total_samples) * 100 if total_samples > 0 else 0

    print(f"Test Loss: {avg_loss:.4f}, Top-1 Accuracy: {accuracy:.2f}%, Top-k Accuracy: {top3_accuracy:.2f}%")
    track_conversation_length(conv_lengths)
    mean_lengths = convmean_per_class(conv_lengths)
    conv_lengths_epochs.append(mean_lengths)
    
    sender.train()
    receiver.train()

    return avg_loss, accuracy, top3_accuracy
   

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

def entropy_from_message(probs):
    """
    Computes binary entropy per sample from sender probabilities.
    Returns a list of scalar tensors, one per sample.
    """
    if isinstance(probs, list):
        probs = torch.stack(probs)  # convert list of tensors to a single tensor

    eps = 1e-8  # for numerical stability
    entropy = - (probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
    entropy_per_sample = entropy.sum(dim=1)  # sum over bits per message
    return [e.unsqueeze(0) for e in entropy_per_sample]






#------------------------------------ Calculate Sender/Receiver Loss --------------------------------------#

def loglikelihood(log_prob, target):
    """
    Computes the log-likelihood for each sample in the batch.

    Args: 
        log_prob (torch.Tensor): Log softmax scores of shape (N, C), 
                                 where N is the batch size and C is the number of classes.
        target (torch.Tensor): Target class indices of shape (N,).
    
    Returns:
        torch.Tensor: Log-likelihood values of shape (N,).
    """
    return log_prob.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
    
        
def calculate_loss_binary(binary_features, binary_probs, logs, baseline_scores, entropy_penalty):
    log_p_z = Variable(binary_features.data) * torch.log(binary_probs + 1e-8) + \
        (1 - Variable(binary_features.data)) * \
        torch.log(1 - binary_probs + 1e-8)
    log_p_z = log_p_z.sum(1)
    weight = Variable(logs.data) - \
        Variable(baseline_scores.clone().detach().data)
    if logs.size(0) > 1:
        weight = weight / np.maximum(1., torch.std(weight.data))
    loss = torch.mean(-1 * weight * log_p_z)

    # Must do both sides of negent, otherwise is skewed towards 0.
    initial_negent = (torch.log(binary_probs + 1e-8)
                      * binary_probs).sum(1).mean()
    inverse_negent = (torch.log((1. - binary_probs) + 1e-8)
                      * (1. - binary_probs)).sum(1).mean()
    negentropy = initial_negent + inverse_negent

    if entropy_penalty is not None:
        loss = (loss + entropy_penalty * negentropy)
    return loss, negentropy.unsqueeze(0)


def multistep_loss_binary(binary_features, binary_probs, logs, baseline_scores, masks, entropy_penalty):
    if masks is not None:
        def mapped_fn(feat, prob, scores, mask, mask_sum):
            if mask_sum == 0:
                return torch.tensor([0.0], device=feat.device), torch.tensor([0.0], device=feat.device)


            feat_size = feat.size()
            prob_size = prob.size()
            logs_size = logs.size()
            scores_size = scores.size()

            feat = feat[mask.expand_as(feat)].view(-1, feat_size[1])
            prob = prob[mask.expand_as(prob)].view(-1, prob_size[1])
            #_logs = logs[mask.expand_as(logs)].view(-1, logs_size[1])
            #print(f"logs_size: {logs_size}, type: {type(logs_size)}")
            #_logs = logs[mask.bool().squeeze()].view(-1, logs_size[0])
            _logs = logs[mask.bool().squeeze()]

            scores = scores[mask.expand_as(scores)].view(-1, scores_size[1])

            return calculate_loss_binary(feat, prob, _logs, scores, entropy_penalty)

        _mask_sums = [m.float().sum().item() for m in masks]

        outp = list(map(mapped_fn, binary_features, binary_probs, baseline_scores, masks, _mask_sums))
        losses = [o[0] for o in outp]
        entropies = [o[1] for o in outp]
        weighted_losses = [l * ms for l, ms in zip(losses, _mask_sums)]
        loss = sum(weighted_losses) / sum(_mask_sums)
    else:
        outp = list(map(lambda feat, prob, scores: calculate_loss_binary(feat, prob, logs, scores, entropy_penalty),
                        binary_features, binary_probs, baseline_scores))
        losses = [o[0] for o in outp]
        entropies = [o[1] for o in outp]
        loss = sum(losses) / len(binary_features)

    return loss, entropies




#------------------------------------ Calculate Baseline Loss --------------------------------------#

def calculate_loss_bas(baseline_scores, logs):
    if not baseline_scores.requires_grad:
        baseline_scores = baseline_scores.clone().detach().requires_grad_(True)
    loss_bas = nn.MSELoss()(baseline_scores, Variable(logs.data))
    return loss_bas


def multistep_loss_bas(baseline_scores, logs, masks):
    if masks is not None:
        losses = map(lambda scores, mask: calculate_loss_bas(
            scores[mask.squeeze()].view(-1, 1), logs[mask.squeeze()].view(-1, 1)),
            baseline_scores, masks)
        _mask_sums = [m.sum().float() for m in masks]
        _losses = [l * ms for l, ms in zip(losses, _mask_sums)]
        loss = sum(_losses) / sum(_mask_sums)
    else:
        losses = map(lambda scores: calculate_loss_bas(scores, logs),
                     baseline_scores)
        loss = sum(losses) / len(baseline_scores)
    return loss




#------------------------------------------Run Training------------------------------------------#

# Define dataset
dataset = SyntheticData(n_distractors,audio_embedding_file=aud_emb,image_embedding_file=img_emb)

# Define split sizes
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#train_dataset = SyntheticData(audio_folder, image_folder,n_distractors)
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

train(sender,receiver,baseline_sen,baseline_rec,train_loader,test_loader,num_epochs=num_epochs,learning_rate=learning_rate,optim_type=optim_type)





#---------------------------------- Eval -------------------------------------------------#

def evaluate_final(sender, receiver, test_loader, baseline_sen, baseline_rec):
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

evaluate_final(sender,receiver,test_loader,baseline_sen,baseline_rec)






# ------------------- Save models + dataset for evaluation ------------------ 
torch.save({
    'sender_state_dict': sender.state_dict(),
    'receiver_state_dict': receiver.state_dict(),
    'baseline_sen_state_dict': baseline_sen.state_dict(),
    'baseline_rec_state_dict': baseline_rec.state_dict()
}, 'models_checkpoint.pth')

# How to load them up later:
# checkpoint = torch.load('models_checkpoint.pth')
# sender.load_state_dict(checkpoint['sender_state_dict'])
# receiver.load_state_dict(checkpoint['receiver_state_dict'])
# baseline_sen.load_state_dict(checkpoint['baseline_sen_state_dict'])
# baseline_rec.load_state_dict(checkpoint['baseline_rec_state_dict'])

# Saving test loader, for evaluation:
with open('test_dataset.pkl', 'wb') as f:
    pickle.dump(test_dataset, f)

# Loading it up later
# with open('test_dataset.pkl', 'rb') as f:
#     test_dataset = pickle.load(f)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

