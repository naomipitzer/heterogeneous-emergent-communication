"""
synthetic_game_data.py

This script generates synthetic audio waveforms for different geometric shape classes 
using distinct waveform generation logic. It embeds these audio clips using the 
VGGish model and optionally applies preprocessing like zero-mean normalization 
or PCA compression.

The output is a `.npz` file containing embeddings and corresponding labels.

---

Arguments:
    --aud_dataset_size   [int]   Total number of audio samples to generate (default: 1200)
    --sr                 [int]   Sample rate in Hz (default: 8000)
    --seconds            [int]   Duration of each audio clip in seconds (default: 2)
    --file_name          [str]   Base name of the output .npz file (suffix added automatically)
    --preprocessing      [str]   Type of preprocessing:
                                 - 'none' → <file_name>.npz
                                 - 'zeromean' → <file_name>-zm.npz
                                 - 'pca' → <file_name>-pca.npz
    --variable           [str]   Type of variability in the dataset:
                                 - 'all' → frequency and amplitude both vary randomly
                                 - 'frequency' → fixed amplitude, frequency classes (dual-label)
                                 - 'amplitude' → fixed frequency, amplitude classes (dual-label)

---

Output Format:
    If --variable=all:
        - embeddings: [N, D] VGGish features
        - labels: shape class indices

    If --variable=frequency or amplitude:
        - embeddings: [N, D] VGGish features
        - shape_labels: shape class indices
        - freq_labels: frequency or amplitude class indices
        - shape_label_map: dict mapping shape name → index
        - freq_label_map: dict mapping class string → index

"""





import argparse
import numpy as np
import os
from tqdm import tqdm
from scipy.signal import square, sawtooth
from torchvggish import vggish, vggish_input
import torch
from sklearn.preprocessing import StandardScaler

# ========== ARGUMENT PARSING ==========
parser = argparse.ArgumentParser()
parser.add_argument('--aud_dataset_size', type=int, default=1200)
parser.add_argument('--sr', type=int, default=8000)
parser.add_argument('--seconds', type=int, default=2)
parser.add_argument('--file_name', type=str, default='vggish_embeddings')
parser.add_argument('--preprocessing', type=str, default='none', choices=['none', 'zeromean', 'pca'])
parser.add_argument('--variable', type=str, default='all', choices=['all', 'frequency', 'amplitude'])
args = parser.parse_args()

# ========== SETUP ==========
sample_rate = args.sr
duration = args.seconds
num_samples = int(sample_rate * duration)
classes = ['circle', 'heart', 'hexagon', 'square', 'star', 'triangle']

# ========== WAVEFORM GENERATORS ==========
def generate_heart_wave(freq, amp):
    t = np.linspace(0, duration, num_samples, endpoint=False)
    carrier = np.sin(2 * np.pi * freq * t)
    modulator = 0.5 * (1 + np.sin(2 * np.pi * freq / 4 * t))
    return amp * carrier * modulator

def generate_hexagon_wave(freq, amp):
    t = np.linspace(0, duration, num_samples, endpoint=False)
    mod = np.sin(2 * np.pi * (freq / 2) * t)
    return amp * np.sin(2 * np.pi * freq * t + mod)

def generate_square_wave(freq, amp):
    t = np.linspace(0, duration, num_samples, endpoint=False)
    return amp * square(2 * np.pi * freq * t)

def generate_star_wave(freq, amp):
    t = np.linspace(0, duration, num_samples, endpoint=False)
    return amp * sawtooth(2 * np.pi * freq * t)

def generate_circle_wave(freq, amp):
    t = np.linspace(0, duration, num_samples, endpoint=False)
    mod = np.sin(2 * np.pi * 2 * t)
    carrier = np.sin(2 * np.pi * freq * (1 + 0.5 * mod) * t)
    return amp * carrier

def generate_triangle_wave(freq, amp):
    t = np.linspace(0, duration, num_samples, endpoint=False)
    stepped_freq = np.piecewise(t,
                                 [t < 0.5, (t >= 0.5) & (t < 1.0), (t >= 1.0) & (t < 1.5), t >= 1.5],
                                 [freq, freq * 0.75, freq * 0.5, freq * 0.25])
    return amp * np.sin(2 * np.pi * stepped_freq * t)

wave_generators = {
    'circle': generate_circle_wave,
    'heart': generate_heart_wave,
    'hexagon': generate_hexagon_wave,
    'square': generate_square_wave,
    'star': generate_star_wave,
    'triangle': generate_triangle_wave
}

# ========== SYNTHETIC DATA GENERATION ==========
def generate_audio_data():
    FIXED_AMPLITUDE = 0.7
    FIXED_FREQUENCY = 440
    NOISE_LEVEL = 0.02
    freq_classes = {'0': (200, 300), '1': (400, 500), '2': (600, 800)}
    amp_classes = {'0': (0.3, 0.4), '1': (0.5, 0.6), '2': (0.7, 0.9)}

    all_audio = []
    all_labels = []

    clips_per_class = args.aud_dataset_size // (len(classes) * (3 if args.variable in ['frequency', 'amplitude'] else 1))

    for shape_label, shape in enumerate(classes):
        generator = wave_generators[shape]

        if args.variable == 'all':
            for _ in range(clips_per_class):
                freq = np.random.uniform(200, 800)
                amp = np.random.uniform(0.3, 0.9)
                waveform = generator(freq, amp)
                noise = np.random.normal(0, NOISE_LEVEL, num_samples)
                waveform = np.clip(waveform + noise, -1.0, 1.0)
                all_audio.append(waveform)
                all_labels.append(shape_label)

        elif args.variable == 'frequency':
            for freq_class, (fmin, fmax) in freq_classes.items():
                for _ in range(clips_per_class):
                    freq = np.random.uniform(fmin, fmax)
                    waveform = generator(freq, FIXED_AMPLITUDE)
                    noise = np.random.normal(0, NOISE_LEVEL, num_samples)
                    waveform = np.clip(waveform + noise, -1.0, 1.0)
                    all_audio.append(waveform)
                    all_labels.append((shape_label, int(freq_class)))

        elif args.variable == 'amplitude':
            for amp_class, (amin, amax) in amp_classes.items():
                for _ in range(clips_per_class):
                    amp = np.random.uniform(amin, amax)
                    waveform = generator(FIXED_FREQUENCY, amp)
                    noise = np.random.normal(0, NOISE_LEVEL, num_samples)
                    waveform = np.clip(waveform + noise, -1.0, 1.0)
                    all_audio.append(waveform)
                    all_labels.append((shape_label, int(amp_class)))

    return all_audio, all_labels

# ========== EMBEDDING WITH VGGISH ==========
audio_model = vggish()
audio_model.eval()

def waveform_to_embedding(waveform):
    from tempfile import NamedTemporaryFile
    from scipy.io.wavfile import write
    with NamedTemporaryFile(suffix='.wav') as tmp_wav:
        write(tmp_wav.name, sample_rate, (waveform * 32767).astype(np.int16))
        example = vggish_input.wavfile_to_examples(tmp_wav.name)
        emb = audio_model(example)
        return emb.detach().numpy().reshape(-1)

# ========== MAIN PIPELINE ==========
print("Generating synthetic audio...")
waveforms, labels = generate_audio_data()

print("Computing embeddings...")
embeddings = []
for wf in tqdm(waveforms):
    try:
        emb = waveform_to_embedding(wf)
        embeddings.append(emb)
    except Exception as e:
        print(f"Error embedding waveform: {e}")

embeddings = np.array(embeddings)

# ========== PREPROCESSING ==========
if args.preprocessing == 'zeromean':
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
elif args.preprocessing == 'pca':
    from sklearn.decomposition import PCA
    embeddings = PCA(n_components=128).fit_transform(embeddings)

# ========== OUTPUT FILENAME LOGIC ==========
suffix_map = {
    'none': '',
    'zeromean': '-zm',
    'pca': '-pca'
}
suffix = suffix_map.get(args.preprocessing, '')
final_filename = args.file_name
if not final_filename.endswith('.npz'):
    final_filename += suffix + '.npz'

# ========== SAVE ==========
if args.variable == 'all':
    out = {
        'embeddings': embeddings,
        'labels': np.array(labels),
    }
else:
    shape_labels = np.array([x[0] for x in labels])
    freq_labels = np.array([x[1] for x in labels])
    shape_label_map = {name: idx for idx, name in enumerate(classes)}
    freq_label_map = {'0': 0, '1': 1, '2': 2}
    out = {
        'embeddings': embeddings,
        'shape_labels': shape_labels,
        'freq_labels': freq_labels,
        'shape_label_map': shape_label_map,
        'freq_label_map': freq_label_map,
    }

np.savez(final_filename, **out)
print(f"Saved embeddings to {final_filename}")