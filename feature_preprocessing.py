import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

training_file = "sequences_training.txt"

train_df = pd.read_csv(training_file, header=None, names=["sequence", "label"])

print("Training Dataset Shape:", train_df.shape)
print("Class Distribution Before Oversampling:")
print(train_df["label"].value_counts())

# Feature generation functions
amino_acids = "ARNDCQEGHILKMFPSTWYV"
dipeptides = [aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids]
physicochemical_properties = {
    'A': [1.8, 0.62], 'R': [-4.5, 1.00], 'N': [-3.5, 0.60], 'D': [-3.5, 0.54],
    'C': [2.5, 0.29], 'Q': [-3.5, 0.39], 'E': [-3.5, 0.37], 'G': [-0.4, 0.48],
    'H': [-3.2, 0.96], 'I': [4.5, 1.38], 'L': [3.8, 1.06], 'K': [-3.9, 1.00],
    'M': [1.9, 0.64], 'F': [2.8, 1.13], 'P': [-1.6, 0.12], 'S': [-0.8, 0.72],
    'T': [-0.7, 0.71], 'W': [-0.9, 1.08], 'Y': [-1.3, 0.63], 'V': [4.2, 1.08]
}

def compute_aac(sequence):
    length = len(sequence)
    counts = {aa: sequence.count(aa) / length for aa in amino_acids}
    return list(counts.values())

def compute_dipeptide_composition(sequence):
    length = len(sequence) - 1
    counts = {dp: 0 for dp in dipeptides}
    for i in range(length):
        dipeptide = sequence[i:i+2]
        if dipeptide in counts:
            counts[dipeptide] += 1
    return [counts[dp] / length if length > 0 else 0 for dp in dipeptides]

def compute_physicochemical_features(sequence):
    num_props = len(next(iter(physicochemical_properties.values())))
    props = np.zeros((len(sequence), num_props))
    for i, aa in enumerate(sequence):
        if aa in physicochemical_properties:
            props[i] = physicochemical_properties[aa]
    return np.concatenate([props.mean(axis=0), props.var(axis=0), props.max(axis=0)])

print("Generating features for the training dataset...")
train_features = np.array([
    np.hstack([
        compute_aac(seq),
        compute_dipeptide_composition(seq),
        compute_physicochemical_features(seq)
    ])
    for seq in train_df["sequence"]
])
train_labels = train_df["label"]

# Scale the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_features = scaler.fit_transform(train_features)

# Step 1: Dimensionality Reduction with PCA 
print("Reducing dimensionality with PCA before applying SMOTE...")
n_components_pre_smote = min(scaled_train_features.shape[1], 100)  # Reduce to 100 components or fewer
pca_pre_smote = PCA(n_components=n_components_pre_smote)
reduced_features_pre_smote = pca_pre_smote.fit_transform(scaled_train_features)
print(f"Features Shape After PCA: {reduced_features_pre_smote.shape}")

# Step 2: Oversample minority classes with SMOTE
print("Handling class imbalance with SMOTE...")
smote = SMOTE(sampling_strategy={'RNA': 2000, 'DNA': 2000, 'DRNA': 2000}, random_state=42)
smote_features, smote_labels = smote.fit_resample(reduced_features_pre_smote, train_labels)
print("After SMOTE:", Counter(smote_labels))

# Step 3: Undersample the majority class
print("Reducing majority class with undersampling...")
rus = RandomUnderSampler(sampling_strategy={'nonDRNA': 3000}, random_state=42)
balanced_features, balanced_labels = rus.fit_resample(smote_features, smote_labels)
print("After Undersampling:", Counter(balanced_labels))

# Step 4: Map labels to integers
label_mapping = {'nonDRNA': 0, 'DNA': 1, 'RNA': 2, 'DRNA': 3}
mapped_labels = [label_mapping[label] for label in balanced_labels]

# Step 5: Combine features and labels into a single dataset
combined_data = np.hstack((balanced_features, np.array(mapped_labels).reshape(-1, 1)))

# Step 6: Save combined dataset to a file
header = [f"feature_{i}" for i in range(balanced_features.shape[1])] + ["label"]
combined_df = pd.DataFrame(combined_data, columns=header)
combined_df.to_csv("final_dataset.csv", index=False)

# Confirm processing
print("Final Dataset with Features and Labels Saved to 'final_dataset.csv'.")
print(f"Final Dataset Shape: {combined_data.shape}")
