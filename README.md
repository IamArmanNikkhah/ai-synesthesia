# ai-synesthesia

**ai‑synesthesia** is a research project that explores multimodal learning models capable of translating between audio and visual modalities—teaching machines to "see" sounds and "hear" visuals. Inspired by the phenomenon of synesthesia in humans, this repository contains code to train, evaluate, and visualize models that map audio signals to images and vice versa.

## Features

- 🎵→🎨 **Audio‑to‑Image Translation** – Generate images conditioned on audio features such as pitch, tempo, and timbre.
- 🎨→🎵 **Image‑to‑Audio Generation** – Synthesize audio sequences based on visual inputs like color palettes, textures, or shapes.
- 🤬 **Multimodal Encoder–Decoder Architecture** – Flexible neural network framework that can plug in different backbones (CNNs, RNNs, Transformers) for encoding audio spectrograms and decoding visuals, or vice versa.
- 🔬 **Evaluation Toolkit** – Scripts for quantitative metrics (e.g. FID, spectral coherence) and qualitative visualization of cross‑modal translations.
- 📊 **Training Pipeline** – End‑to‑end training scripts with configurable hyperparameters, checkpointing, and support for GPU acceleration.

## Project Structure

- `model/` – Model definitions and architectures for encoders/decoders.
- `training/` – Training loops, dataset loaders, configuration files.
- `evaluation/` – Evaluation scripts and metrics for both modalities.
- `utils/` – Utility functions for preprocessing, logging, and visualization.
- `pipeline_full.py` – Example script to run the full synesthesia pipeline from data loading through training to evaluation.

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/IamArmanNikkhah/ai-synesthesia.git
   cd ai-synesthesia
   ```

2. **Install dependencies**

   It's recommended to use Python 3.9+ with `pip` and virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**

   - Place your paired audio–image dataset in a directory structure similar to:

     ```
     data/
       train/
         audio/
         images/
       val/
         audio/
         images/
     ```
   - Supported audio formats: `.wav`, `.mp3`. Images should be `.png` or `.jpg`.

4. **Train a model**

   To train the default audio‑to‑image model:

   ```bash
   python training/train.py \
       --config configs/audio2img.yaml \
       --data_dir data/train \
       --val_dir data/val \
       --epochs 100 \
       --batch_size 16
   ```

   For image‑to‑audio, use the corresponding config file (`configs/img2audio.yaml`).

5. **Evaluate and visualize**

   ```bash
   python evaluation/evaluate_model.py --checkpoint path/to/checkpoint.pth --output_dir outputs
   ```

   Generated images and audio will be saved in the `outputs` directory for inspection.

## Contributing

Contributions are welcome! If you have ideas to improve the architecture, add new datasets, or implement additional cross‑modal tasks:

- Fork the repository and create a new branch: `git checkout -b feature/your-feature`.
- Commit your changes and push your branch.
- Open a pull request describing your contributions.

Please follow standard Python coding conventions and document any new modules.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
