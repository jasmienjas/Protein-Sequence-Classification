# Protein Sequence Classification

This project involves processing and transforming protein sequence data for classification using Support Vector Machines (SVM). The workflow addresses class imbalance, high dimensionality, and the need for biologically relevant feature engineering.

---

## Feature Engineering
Computed three types of features to represent protein sequences:

1. **Amino Acid Composition (AAC)**:
   - Frequency of each amino acid in a sequence.

2. **Dipeptide Composition**:
   - Frequency of consecutive amino acid pairs.

3. **Physicochemical Properties**:
   - Statistical properties (mean, variance, max) of amino acid attributes.

**Justification**:  
These features capture the biological and chemical characteristics of protein sequences, converting them into fixed-length numerical vectors suitable for machine learning.

---

## Scaling Features
- **Method**: Min-Max Scaling to the range [0, 1].  
- **Justification**:  
  Machine learning models, especially SVM with RBF or polynomial kernels, perform better when features are normalized to a comparable scale.

---

## Dimensionality Reduction
- **Method**: Principal Component Analysis (PCA) to reduce features to a maximum of 100 dimensions.  
- **Justification**:  
  - High-dimensional datasets (e.g., 426 features) increase memory usage and computational load.
  - PCA reduces redundancy, retains most of the variance, and makes the dataset smaller and more efficient for oversampling.

---

## Handling Class Imbalance

### 1. Oversampling Minority Classes with SMOTE
- **Method**: Applied SMOTE to generate synthetic samples for minority classes (`RNA`, `DNA`, `DRNA`) to reach 2,000 samples each.  
- **Justification**:  
  - Class imbalance can degrade model performance on underrepresented classes.
  - SMOTE improves representation of minority classes by generating synthetic samples through interpolation.

### 2. Undersampling the Majority Class
- **Method**: Reduced the majority class (`nonDRNA`) to 3,000 samples using Random Undersampling.  
- **Justification**:  
  - Prevents the dataset from becoming unnecessarily large while maintaining balance.
  - Ensures that the majority class does not dominate the model’s decision-making.

---

## Mapping Labels to Integers
- Converted class labels into integers for compatibility with machine learning algorithms:
  - `nonDRNA → 0`
  - `DNA → 1`
  - `RNA → 2`
  - `DRNA → 3`

---

## Final Dataset
- **Features**: The final dataset contains PCA-reduced features combined with integer-mapped labels.
- **Structure**:  
  - Features: `feature_0`, `feature_1`, ..., `feature_99`  
  - Labels: `label`  
- **File**: The final dataset is saved as `final_dataset.csv` for downstream training and evaluation.

---

## Workflow Summary
1. **Feature Engineering**: AAC, dipeptide composition, and physicochemical properties.  
2. **Scaling**: Min-Max Scaling to normalize feature ranges.  
3. **Dimensionality Reduction**: PCA to 100 dimensions for efficiency.  
4. **Oversampling**: SMOTE to balance minority classes (`RNA`, `DNA`, `DRNA`).  
5. **Undersampling**: Reduced `nonDRNA` samples to 3,000.  
6. **Label Mapping**: Converted class labels to integers for compatibility.  
7. **Dataset Saving**: Final dataset saved as `final_dataset.csv`.

---

## Project Goals
This workflow ensures:
1. **Balanced Representation**: Minority classes are well-represented for better classification performance.
2. **Scalability**: Dimensionality reduction and undersampling make the dataset computationally efficient.
3. **Biological Relevance**: Features capture meaningful biological and chemical properties of protein sequences.
