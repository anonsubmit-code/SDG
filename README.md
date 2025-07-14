# Synthetic Linux Log Generation and Evaluation

This repository contains code, datasets, and evaluation metrics for generating and analyzing synthetic Linux log data using various generative AI models including CTGAN, GPT-2, GPT-3.5-Turbo, GPT-4.1-Mini, GPT-o4-Mini, and LSTM.

## 📁 Project Structure

```plaintext
├── 0_Real_and_Synthetic_Datasets/
├── 1_Generating_Synthetic_Data/
├── 2_Statistical_Results_from_All_Models/
├── 3_Statistical_Analysis/
├── 4_Statistical_Data_to_Graph/
└── Results_in_Graph/

## 📁 Folder Details

### 🔹 `0_Real_and_Synthetic_Datasets/`
Contains both real and synthetic Linux kernel log datasets. Subdirectories are organized by model type and include various sample sizes (e.g., 1K, 10K, 100K).

- **Real datasets**: Raw logs captured from LTTng.
- **Synthetic datasets** generated using:
  - CTGAN  
  - LSTM  
  - GPT-2  
  - GPT-3.5-Turbo  
  - GPT-4.1-Mini  
  - GPT-o4-Mini  

---

### 🔹 `1_Generating_Synthetic_Data/`
Includes all scripts and model configurations used to generate synthetic data.

- Preprocessing and formatting scripts for real logs  
- Training scripts for CTGAN, LSTM, GPT-2, and Few-Shot synthetic data generation using GPT-3.5/4.1/o4 models  
- Sampling scripts to generate synthetic logs  
- Model configuration files and checkpoints  

---

### 🔹 `2_Statistical_Results_from_All_Models/`
Holds raw statistical outputs collected after evaluating each model.

- **Fidelity**: Wasserstein distance, distributional comparisons  
- **Utility**: Classifier accuracy for distinguishing real vs. synthetic  
- **Privacy**

Metrics are saved as .text files.

---

### 🔹 `3_Statistical_Analysis/`
Analyzes and summarizes raw metrics from all models.

- Scripts for computing descriptive statistics (mean, std, etc.)  
- Metric-wise comparison between models  
- Intermediate outputs for graph plotting  

---

### 🔹 `4_Statistical_Data_to_Graph/`
Stores data structured for visualization.

- Aggregated metrics grouped by model and sample size  
- CSVs ready to be used with plotting libraries (e.g., Matplotlib, Seaborn)  
- Input files for the final graphs  

---

### 🔹 `Results_in_Graph/`
Final visualizations derived from the statistical analysis.

- Bar charts, line graphs, and comparison plots  
- Metrics visualized: fidelity, utility, privacy, range/category completeness  
