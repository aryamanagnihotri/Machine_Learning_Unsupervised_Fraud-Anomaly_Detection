import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
#Output results
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)
#Loading dataset
df = pd.read_csv("creditcard.csv")
print("Shape:", df.shape)
print(df.head())
#checking missing and dups
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
#classes info
counts = df["Class"].value_counts()
print("Class counts:\n", counts.to_string())

print(f"Fraud rate: {counts[1] / len(df):.4%})")
print("Class Ratio: (Normal: Fraud)")
print(f"{counts[0] / counts[1]:.0f}:1")
#inspecting class imbalance
print("\nFeature scales (Time/Amount vs PCA components):")
print(df.drop(columns="Class").describe().T[["mean", "std", "min", "max"]].round(3).to_string())
#plotting class imbalance
#Plot 1: Normal v/s Fraud class
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
counts.plot.bar(ax=ax[0], color=["Blue", "Red"], logy=True)
ax[0].set_title("Class Distribution of Transactions (Normal vs Fraud)")
ax[0].set_xlabel("Transaction class")
ax[0].set_ylabel("Number of transactions (log scale)")
ax[0].set_xticklabels(["Normal (0)", "Fraud (1)"], rotation=0)
for i, v in enumerate(counts.values):                 
    ax[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
 
# Plot 2: Amount distribution
ax[1].hist(np.log1p(df["Amount"]), bins=60, color="Green", edgecolor="white")
ax[1].set_title("Distribution of Transaction Amount (log-transformed)")
ax[1].set_xlabel("log(1 + Amount)")
ax[1].set_ylabel("Number of transactions")
 
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/01_data_inspection.png", dpi=130); plt.close()

#preprocessing: removing dups and dropping cols(class and time)
df = df.drop_duplicates().reset_index(drop=True)      
X = df.drop(columns=["Class", "Time"]).copy()          
X["Amount"] = np.log1p(X["Amount"])                    
y = df["Class"].values

#test/train/validation split(3:1:1)
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

#standart scaler on training data
scaler = RobustScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
#Data Split info
print(f"\nTrain {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
print(f"Frauds -> train {y_train.sum()}, val {y_val.sum()}, test {y_test.sum()}")
