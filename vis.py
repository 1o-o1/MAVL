# Re-import necessary libraries since execution state was reset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Creating synthetic data similar to the given chart, but for CheXpert and MedTX datasets
aspects = ["desc.", "opacity", "border", "fluid", "location", "shape", "pattern", "texture"]
num_aspects = list(range(1, 8))

# Generating sample AUC scores for CheXpert and MedTX datasets
auc_chexpert = np.linspace(85, 90, len(aspects))
auc_medtx = np.linspace(70, 80, len(aspects))

# Creating DataFrame for aspect-wise AUC scores
df_aspects = pd.DataFrame({
    "Aspect": aspects,
    "CheXpert AUC": auc_chexpert,
    "MedTX AUC": auc_medtx
})

# Generating sample AUC scores for increasing number of added visual aspects
auc_chexpert_incremental = np.linspace(85, 90, len(num_aspects))
auc_medtx_incremental = np.linspace(70, 80, len(num_aspects))

# Creating DataFrame for incremental aspect addition
df_incremental = pd.DataFrame({
    "Number of Added Aspects": num_aspects,
    "CheXpert AUC": auc_chexpert_incremental,
    "MedTX AUC": auc_medtx_incremental
})

# Display tables
import ace_tools as tools
tools.display_dataframe_to_user(name="Aspect-wise AUC Scores", dataframe=df_aspects)
tools.display_dataframe_to_user(name="Incremental Aspect Addition AUC Scores", dataframe=df_incremental)

# Plotting the data
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for aspect-wise AUC scores
axes[0].bar(aspects, auc_medtx, color='red', label="MedTX")
axes[0].bar(aspects, auc_chexpert, color='green', alpha=0.6, label="CheXpert")
axes[0].set_ylabel("AUC Scores")
axes[0].set_xlabel("Added Aspect")
axes[0].set_title("(a) Aspect-wise AUC Scores")
axes[0].legend()

# Plot for number of added aspects
axes[1].bar(num_aspects, auc_medtx_incremental, color='red', label="MedTX")
axes[1].bar(num_aspects, auc_chexpert_incremental, color='green', alpha=0.6, label="CheXpert")
axes[1].set_ylabel("AUC Scores")
axes[1].set_xlabel("Number of Added Visual Aspects")
axes[1].set_title("(b) Incremental Aspect Addition")
axes[1].legend()

plt.tight_layout()
plt.show()
