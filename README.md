# Quantum Machine Learning Implementations with Qiskit

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.2.4-purple.svg)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive framework for comparing **Quantum Machine Learning (QML)** models against classical machine learning algorithms on healthcare prediction tasks. This repository implements state-of-the-art quantum algorithms using Qiskit 1.0+ and provides automated benchmarking capabilities for both simulated and real quantum hardware.

## 🎯 Project Overview

This project serves as a complete comparison framework for evaluating quantum machine learning models against classical baselines on real-world healthcare datasets. The implementation supports:

- **Multiple Quantum Algorithms**: VQC, PegasosQSVC, EstimatorQNN, SamplerQNN
- **Classical Baselines**: Logistic Regression, SVM, Naive Bayes, K-NN
- **Real & Simulated Quantum Hardware**: IBM Quantum integration with automatic backend selection
- **Comprehensive Evaluation**: Cross-validation, multiple metrics, execution time tracking
- **Automated Reporting**: Visualization, results export, and performance comparison tables

## ✨ Key Features

### 🔬 **Advanced Model Comparison**
- **5 Classical Models**: Logistic Regression, Naive Bayes, SVM (Linear/RBF), K-Nearest Neighbors
- **4 Quantum Models**: Variational Quantum Classifier (VQC), PegasosQSVC, EstimatorQNN, SamplerQNN
- **Dual Execution**: Both simulated quantum hardware and real IBM Quantum devices

### 📊 **Intelligent Data Processing**
- **Automatic Preprocessing**: Missing value handling, categorical encoding, feature scaling
- **Class Imbalance Handling**: SMOTE oversampling and random undersampling
- **Feature Selection**: Correlation-based feature selection with configurable counts
- **Cross-Validation**: Stratified K-fold validation with configurable splits

### 📈 **Comprehensive Evaluation & Reporting**
- **Multiple Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Performance Tracking**: Execution time measurement for all models
- **Automated Visualization**: Comparison charts, performance tables, styled reports
- **Export Capabilities**: JSON results, CSV metrics, high-resolution plots

### ⚙️ **Flexible Configuration**
- **Easy Dataset Switching**: Simple constant modification for different datasets
- **Configurable Parameters**: Feature counts, CV folds, optimization iterations
- **Output Customization**: Plot formats, file naming, directory structure
- **Quantum Backend Selection**: Automatic least-busy backend selection or manual specification

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook/Lab
- IBM Quantum account (for real quantum hardware access)

### 1. Clone the Repository
```bash
git clone https://github.com/asterios-pantousas/Qiskit_QML_Implementations.git
cd Qiskit_QML_Implementations
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. IBM Quantum Setup (Optional - for real quantum hardware)
1. Create a free account at [IBM Quantum](https://quantum-computing.ibm.com/)
2. Get your API token from your account dashboard
3. Set up environment variable:
```bash
export MY_IBM_QUANTUM_TOKEN="your_token_here"
```
Or create a `.env` file:
```
MY_IBM_QUANTUM_TOKEN=your_token_here
```

### 4. Verify Installation
```bash
python -c "import qiskit; print(qiskit.__version__)"
```

## 📋 Configuration Guide

The main configuration is handled through constants at the top of `generic_comparator.ipynb`:

### Dataset Configuration
```python
DATASET_PATH = "datasets/heart.csv"        # Path to your CSV dataset
TARGET_COLUMN = "HeartDisease"             # Target column name
FEATURE_COUNTS = [10]                      # List of feature counts to evaluate
```

### Experiment Parameters
```python
CROSS_VALIDATION_FOLDS = 2                # Number of CV folds
HANDLE_IMBALANCE = 'smote'                 # 'smote', 'undersample', or None
IMBALANCE_THRESHOLD = 0.7                  # Threshold for imbalance detection
USE_FAKE_QUANTUM = True                    # True for simulation, False for real hardware
VERBOSE = True                             # Detailed output during execution
```

### Output Configuration
```python
# Plot Settings
SAVE_PLOTS = True                          # Enable plot saving
PLOT_DIR = "plots"                         # Output directory
PLOT_FORMAT = "png"                        # File format (png, pdf, svg)
PLOT_DPI = 300                             # Resolution

# Results Settings
SAVE_DICT = True                           # Enable results saving
DICT_DIR = "results"                       # Results directory
INCLUDE_TIMESTAMP = True                   # Add timestamp to filenames
```

## 🎮 Usage Examples

### Basic Usage - Heart Disease Prediction
```python
# Configure for heart disease dataset
DATASET_PATH = "datasets/heart.csv"
TARGET_COLUMN = "HeartDisease"
FEATURE_COUNTS = [5, 10]

# Run the comparison
all_results = main()
```

### Stroke Prediction with Custom Settings
```python
# Configure for stroke dataset
DATASET_PATH = "datasets/stroke.csv"
TARGET_COLUMN = "stroke"
FEATURE_COUNTS = [6, 10]
CROSS_VALIDATION_FOLDS = 3
HANDLE_IMBALANCE = 'smote'

# Run the comparison
all_results = main()
```

### Real Quantum Hardware Execution
```python
# Enable real quantum hardware
USE_FAKE_QUANTUM = False
MY_IBM_TOKEN = "your_token_here"

# Run with quantum hardware
all_results = main()
```

## 🤖 Model Descriptions

### Classical Models
- **Logistic Regression**: Linear probabilistic classifier
- **Naive Bayes**: Gaussian Naive Bayes classifier
- **SVM (Linear)**: Support Vector Machine with linear kernel
- **SVM (RBF)**: Support Vector Machine with radial basis function kernel
- **K-Nearest Neighbors**: Instance-based learning algorithm

### Quantum Models
- **VQC (Variational Quantum Classifier)**: Parameterized quantum circuit with classical optimization
- **PegasosQSVC**: Quantum kernel-based SVM using fidelity quantum kernel
- **EstimatorQNN**: Quantum Neural Network using expectation values
- **SamplerQNN**: Quantum Neural Network using measurement probabilities

### Quantum Circuit Components
- **Feature Maps**: ZZFeatureMap, ZFeatureMap for data encoding
- **Ansätze**: RealAmplitudes variational form
- **Optimizers**: COBYLA (Constrained Optimization BY Linear Approximation)

## 📊 Output & Results

### Generated Files
```
results/
├── {dataset}_{features}_model_results_{timestamp}.txt    # JSON results
plots/
├── {dataset}_accuracy_comparison.png                     # Performance charts
├── {dataset}_precision_table.png                         # Styled tables
├── {dataset}_accuracy_score_{features}.csv              # Raw metrics
└── ...
```

### Results Structure
```json
{
  "10": {
    "Logistic Regression": [
      {"accuracy": 0.8500},
      {"precision": 0.8200},
      {"recall": 0.8100},
      {"f1-score": 0.8150},
      {"Elapsed Time": 0.045}
    ],
    "Quantum VQC": [...],
    ...
  }
}
```

### Performance Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Execution Time**: Model training and prediction time

## 📁 Project Structure

```
Qiskit_QML_Implementations/
├── generic_comparator.ipynb          # Main comparison framework
├── generic_comparator.py             # Python script version
├── requirements.txt                  # Dependencies
├── README.md                         # This file
├── datasets/                         # Healthcare datasets
│   ├── heart.csv                     # Heart disease data
│   ├── stroke.csv                    # Stroke prediction data
│   ├── diabetes.csv                  # Diabetes data
│   └── README.md                     # Dataset documentation
├── plots/                            # Generated visualizations
├── results/                          # Experiment results
└── docs/                            # Additional documentation
```

## 📚 Datasets

### Heart Disease Prediction
- **Source**: [Kaggle Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Features**: 11 clinical features (age, cholesterol, blood pressure, etc.)
- **Target**: Binary classification (heart disease presence)

### Stroke Prediction
- **Source**: [Kaggle Stroke Prediction](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Features**: 10 demographic and health features
- **Target**: Binary classification (stroke occurrence)

### Diabetes Prediction
- **Source**: [Kaggle Diabetes Prediction](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- **Features**: Various health indicators
- **Target**: Binary classification (diabetes presence)

## 🔧 Requirements

### Core Dependencies
- **qiskit**: 1.2.4+ (Quantum computing framework)
- **qiskit-machine-learning**: 0.7.2+ (QML algorithms)
- **qiskit-ibm-runtime**: 0.23.0+ (IBM Quantum access)
- **scikit-learn**: 1.4.2+ (Classical ML algorithms)
- **pandas**: 2.2.2+ (Data manipulation)
- **numpy**: 1.26.4+ (Numerical computing)
- **matplotlib**: 3.8.4+ (Plotting)
- **imbalanced-learn**: 0.13.0+ (Class imbalance handling)

### Optional Dependencies
- **dataframe_image**: For styled table exports
- **jupyter**: For notebook interface
- **plotly**: For interactive visualizations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IBM Quantum** for providing quantum computing resources
- **Qiskit Community** for the excellent quantum machine learning tutorials
- **Kaggle** for providing the healthcare datasets
- **Scikit-learn** for classical machine learning implementations

## 📞 Contact

**Asterios Pantousas** - [GitHub](https://github.com/asterios-pantousas) - [LinkedIn](https://www.linkedin.com/in/apantousas/)

Project Link: [https://github.com/asterios-pantousas/Qiskit_QML_Implementations](https://github.com/asterios-pantousas/Qiskit_QML_Implementations)

---

⭐ **Star this repository if you find it helpful!**
