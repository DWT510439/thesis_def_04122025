import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os

# =========================
# CONFIGURATION
# =========================
DATA_PATH = r"C:\Users\DouweTerlouw\thesis\results\optuna_plots_10_fewshot_50_trials_2110\trial_results_log.csv"
TARGET_COL = 'nDCG@5'
# =========================

# --- STEP 1: Load data ---
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded data from: {DATA_PATH}")
print(f"Shape: {df.shape}")

# --- STEP 2: Select Bayesian hyperparameter columns ---
bayesian_vars = [
    'lr', 'epochs', 'batch_size', 'warmup_ratio',
    'scheduler', 'gradient_accumulation_steps', 'dropout', 'lora_rank'
]
bayesian_vars = [col for col in bayesian_vars if col in df.columns]

if TARGET_COL not in df.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not found in CSV")

df = df.dropna(subset=bayesian_vars + [TARGET_COL])

# --- STEP 3: Separate continuous vs discrete variables ---
continuous_vars = [col for col in ['lr', 'warmup_ratio', 'dropout'] if col in df.columns]
discrete_vars = [col for col in ['epochs', 'batch_size', 'gradient_accumulation_steps', 'lora_rank', 'scheduler'] if col in df.columns]

print("\n📊 Continuous variables:", continuous_vars)
print("📊 Discrete variables:", discrete_vars)

# --- STEP 4: Preprocessing pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ("continuous", StandardScaler(), continuous_vars),
        ("discrete", OneHotEncoder(drop='first', handle_unknown='ignore'), discrete_vars)
    ]
)

# --- STEP 5: PCA pipeline (hybrid setup) ---

# First: fit a full PCA for the elbow plot
pca_full = PCA().fit(preprocessor.fit_transform(df[bayesian_vars]))

# --- Show explained variance and elbow plot ---
print("\nExplained variance ratio by each principal component:")
for i, ratio in enumerate(pca_full.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.2%}")

cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='o')
plt.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance by PCA Components")
plt.legend()
plt.grid(True)
plt.show()

# --- Select number of components based on elbow or threshold ---
# Option 1: fixed number (e.g., 3)
# n_components = 3

# Option 2: keep enough to explain 90% variance automatically
n_components = 0.9

print(f"\nUsing n_components = {n_components} for final PCA.\n")

# --- Run PCA again with the chosen number of components ---
pca = PCA(n_components=n_components)
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("pca", pca)])
pca_components = pipeline.fit_transform(df[bayesian_vars])

# --- STEP 6: Combine PCA output with performance metric ---
pca_df = pd.DataFrame(
    pca_components, 
    columns=[f"PC{i+1}" for i in range(pca_components.shape[1])]
)
pca_df[TARGET_COL] = df[TARGET_COL].values

# --- STEP 9: PCA loadings (variable contributions) ---
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
loadings = pd.DataFrame(
    pca.components_.T,
    index=feature_names,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)]
)
print("\n🔍 PCA Loadings (importance of each variable in each component):")
print(loadings.round(3))

# --- STEP 10: Correlation between PCs and performance metric ---
corr = pca_df.corr()[[TARGET_COL]].sort_values(by=TARGET_COL, ascending=False)
print("\n📈 Correlation of each Principal Component with performance (nDCG@5):")
print(corr.round(3))

# # --- STEP 11: Bar plot of variable loadings for each component ---
# num_components_to_plot = min(4, pca.n_components_)  # plot first 4 or fewer components
# for i in range(num_components_to_plot):
#     plt.figure(figsize=(10, 4))
#     component = f"PC{i+1}"
#     sorted_loadings = loadings[component].sort_values(ascending=False)
#     plt.bar(sorted_loadings.index, sorted_loadings.values)
#     plt.xticks(rotation=90)
#     plt.title(f"Variable Loadings for {component}")
#     plt.ylabel("Loading Weight")
#     plt.grid(axis="y", linestyle="--", alpha=0.6)
#     plt.tight_layout()
#     plt.show()

# # --- STEP 12: Scatter plot of first two PCs vs performance ---
# plt.figure(figsize=(8, 6))
# plt.scatter(pca_df["PC1"], pca_df["PC2"], c=pca_df[TARGET_COL], cmap="viridis", s=80, edgecolor="k")
# plt.colorbar(label=TARGET_COL)
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title("PCA of Bayesian Hyperparameters vs Performance (nDCG@5)")
# plt.show()

# --- STEP 13: Plot correlation of PCs with performance ---
plt.figure(figsize=(8, 4))
plt.bar(corr.index[:-1], corr[TARGET_COL][:-1])
plt.xticks(rotation=45)
plt.ylabel("Correlation with nDCG@5")
plt.title("Principal Components vs Performance")
plt.show()
