# AWAD

AWAD is a Zero-shot Anomaly Detection (ZSAD) framework built upon the AnomalyCLIP architecture. This version introduces **Orthogonal Regularization** and **AAG-Pooling** to enhance feature decorrelation and cross-domain generalization.

## 🛠️ Installation

Ensure you have Python 3.8+ and install the required dependencies:

```bash
pip install -r requirements.txt

🚀 Quick Start
1. Training
bash train.sh

Logs and checkpoints will be saved in the checkpoints/ directory with the _ortho_aag suffix.
2. Testing
To evaluate the model in a zero-shot setting on target datasets, run:
bash test.sh
✨ Key Features
AAG-Pooling: Improved feature aggregation for better localization of multi-scale anomalies.

Orthogonal Regularization: Constrains the feature space during training to reduce redundancy and improve zero-shot performance.

Object-Agnostic Learning: Utilizes dual-path attention and learnable text tokens to separate anomaly characteristics from object semantics.

Multi-Scale Extraction: Extracts features from intermediate Transformer layers (6, 12, 18, 24) for comprehensive anomaly detection.
📊 Dataset Preparation
The project supports various industrial and medical datasets (MVTec, VisA, SDD, etc.).

Place your datasets in the designated directory.

Generate the necessary metadata using scripts in generate_dataset_json/.
