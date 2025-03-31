# GACL-Net
GACL-Net: Hybrid Deep Learning Framework for Accurate Motor Imagery Classification in Stroke Rehabilitation

This repository contains the implementation of graph-attentive convolutional long short-term memory network (GACL-Net), a novel deep learning model designed for accurate motor imagery (MI) classification in stroke rehabilitation. GACL-Net integrates multi-scale convolutional layers, attention fusion mechanisms, graph convolutional networks, and bidirectional LSTMs to enhance classification robustness and generalization across stroke patients.

## Project Structure
This repository includes scripts for feature extraction, feature selection, model definition, training, and statistical analysis:

**1. Feature Extraction (FeaturesExtraction.py)**

Extracts spatial, temporal, and spectral features from EEG signals, including:
- Alpha & Beta Band Power
- Hilbert Amplitude Envelope
- EEG Coherence
- Event-Related Desynchronization (ERD/ERS)
- Fractal Dimension & Lyapunov Exponent
  
**2. Feature Selection (GA_Fetures_Selection.py)**

Implements genetic algorithm (GA) for optimal feature subset selection, reducing model complexity while maintaining high accuracy.

**3. Model Definition (GACL_Model.py)**

Defines the GACL-Net architecture, including:
- Multi-scale convolutional block
- Attention fusion layer
- Graph convolutional layer
- Bidirectional LSTM with attention
- Hierarchical feature aggregation & dense layers
  
**4. Model Training & Evaluation (Model_Training.py)**
- Loads extracted features and applies data augmentation & normalization.
- Splits the dataset into training, validation, and test sets.
- Trains GACL-Net with cross-entropy loss & Adam optimizer.
- Evaluates accuracy, precision, recall, and F1-score.
  
**5. Statistical Analysis (Statistical_Analysis.py)**
Performs ANOVA-based statistical analysis on EEG variability across stroke patients.

# If you find this work useful, please cite our article:
Bunterngchit, C., Baniata, L.H., Baniata, M.H., ALDabbas, A., Khair, M.A., Chearanai, T. & Kang, S. (2025). GACL-Net: Hybrid Deep Learning Framework for Accurate Motor Imagery Classification in Stroke Rehabilitation. Computers, Materials & Continua. 83(1). 517-536. https://doi.org/10.32604/cmc.2025.060368

# Publicly available datasets used in the article:

Dataset 1 from Liu et al.: https://www.nature.com/articles/s41597-023-02787-8

Dataset link: https://figshare.com/articles/dataset/EEG_datasets_of_stroke_patients/21679035/5

Dataset 2 from Tianyu Jia, Dataset link: https://figshare.com/articles/dataset/EEG_data_of_motor_imagery_for_stroke_patients/7636301
