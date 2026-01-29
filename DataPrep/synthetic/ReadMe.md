# Generating Synthetic Data - "Shapes World"

[Part III Project](https://github.com/naomipitzer/comp3200-p3project/blob/main/Report/Dissertation.pdf) experiments were initially conducted using a synthetic dataset called "Shapes World". This dataset can be generated and embeddings directly extracted into a _.npz_ file using procedures detailed in the sections below.

This repository contains both **Jupyter notebooks** and **automated Python scripts** for:
- Generating synthetic audio or image datasets
- Extracting embeddings (via VGGish or VGG16)
- Fine-tuning VGG-16 model on generated data
- Applying preprocessing such as zero-mean normalisation or PCA

Notebooks can be used interactively for exploration, or `.py` scripts can be run for fully automated dataset and embedding generation.

More details on the dataset can be found in the [project report](https://github.com/naomipitzer/comp3200-p3project/blob/main/Report/Dissertation.pdf).

---
## Audio Generation

This *audio_npz.py* script generates synthetic audio waveforms, extracts embeddings using the VGGish model, and saves them with optional preprocessing.

### Features

- Supports three preprocessing modes:
  - **none**: raw embeddings saved as `<file_name>.npz`
  - **zeromean**: zero-mean normalized embeddings saved as `<file_name>-zm.npz`
  - **pca**: PCA-compressed embeddings saved as `<file_name>-pca.npz`
- Supports three dataset variability options:
  - **all**: amplitude and frequency vary, single label output
  - **frequency**: fixed amplitude and noise, frequency varies, dual labels output (used for [Section 5.3.3](https://github.com/naomipitzer/comp3200-p3project/blob/main/Report/Dissertation.pdf#page=37) ablation experiment)
  - **amplitude**: fixed frequency and noise, amplitude varies, dual labels output (used for [Section 5.3.3](https://github.com/naomipitzer/comp3200-p3project/blob/main/Report/Dissertation.pdf#page=37) ablation experiment)

### Arguments
| Argument          | Description                                     | Default   |
|-------------------|------------------------------------------------|-----------|
| `--aud_dataset_size` | Number of audio samples to generate            | 1200      |
| `--sr`             | Sample rate in Hz                               | 8000      |
| `--seconds`        | Duration of each audio clip in seconds         | 2         |
| `--file_name`      | Base filename for the output `.npz` file       | `embeddings` |
| `--preprocessing`  | Preprocessing mode: `none`, `zeromean`, `pca`  | `none`    |
| `--variable`       | Dataset variability: `all`, `frequency`, `amplitude` | `all`     |


### Output Structure

- For `--variable=all`:
  - `embeddings`: VGGish features array `[N, D]`
  - `labels`: shape class indices

- For `--variable=frequency` or `--variable=amplitude`:
  - `embeddings`: VGGish features array `[N, D]`
  - `shape_labels`: shape class indices
  - `freq_labels`: frequency or amplitude class indices
  - `shape_label_map`: dictionary mapping shape names to indices
  - `freq_label_map`: dictionary mapping frequency/amplitude classes to indices


### Example Usage

```bash
python audio_npz.py \
    --aud_dataset_size=1200 \
    --sr=8000 \
    --seconds=2 \
    --file_name=vgg_embeddings \
    --preprocessing=zeromean \
    --variable=frequency
```

---

## Image Generation

This *images_npz.py* script generates synthetic shape images, fine-tunes (or loads) a VGG16 model on the shapes, extracts embeddings from images, and saves them with optional preprocessing.

### Features

- Automatically fine-tunes the VGG16 model unless a pre-trained model file (`vgg16-finetuned.pth`) is found.
- Generates synthetic shape images of six classes: circle, triangle, square, hexagon, star, and heart.
- Extracts embeddings of size 128 from the VGG16 classifier layer.
- Supports two preprocessing modes for embeddings:
  - **none**: raw embeddings saved as `<file_name>.npz`
  - **zeromean**: zero-mean normalized embeddings saved as `<file_name>-zm.npz`
- Allows configuration of dataset size, image size, and output file name.

### Arguments

| Argument             | Description                                      | Default        |
|----------------------|-------------------------------------------------|----------------|
| `--instances_per_class` | Number of images per shape class generated      | 400            |
| `--image_size`       | Width and height of generated images (pixels)   | 256            |
| `--file_name`        | Base filename for the output `.npz` file         | `vgg_image_embeddings` |
| `--preprocessing`    | Embedding preprocessing: `none`, `zeromean`      | `none`         |
| `--save_vgg16`       | Whether to save the fine-tuned VGG16 model (True/False) | `False`     |

### Output Structure

- `embeddings`: numpy array of shape `[N, 128]` containing image embeddings
- `labels`: numpy array of class indices corresponding to each embedding
- `label_map`: dictionary mapping shape names (circle, triangle, etc.) to numeric indices

If `--preprocessing=zeromean`, zero-mean normalized embeddings are saved separately with suffix `-zm.npz`.

### Usage Example

Generate images, fine-tune the model (if needed), extract embeddings, and save zero-meaned embeddings:

```bash
python images_npz.py \
  --instances_per_class=400 \
  --image_size=256 \
  --file_name=vgg_image_embeddings \
  --preprocessing=zeromean \
  --save_vgg16=False
```