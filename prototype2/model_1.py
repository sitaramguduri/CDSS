import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import load_npz
import random
import random


# -----------------------------
# Load data (match scoring_2.ipynb)
# -----------------------------
Gdrug_mu = np.load("Gdrug_mu_restricted.npy")
Gdrug_sigma = np.load("Gdrug_sigma_restricted.npy")
Gadr = load_npz("Gadr_restricted.npz")

eps = 1e-6
Gdrug_eff = Gdrug_mu / (Gdrug_sigma + eps)

num_drugs, num_genes = Gdrug_eff.shape
num_adrs = Gadr.shape[0]

print("Gdrug_eff shape:", Gdrug_eff.shape)
print("Gadr shape:", Gadr.shape)

# -----------------------------
# Load SIDER positives
# -----------------------------
sider_df = pd.read_csv("sider_lincs_common_clean_FINAL.csv")

# Robust column detection
# -----------------------------
# Load index mappings
# -----------------------------
drug_index_df = pd.read_csv("drug_index.csv")
adr_index_df = pd.read_csv("adr_index.csv")
print("Drug index columns:", drug_index_df.columns.tolist())
print("ADR index columns:", adr_index_df.columns.tolist())
drug_id_to_idx = dict(zip(drug_index_df["drug_id"], drug_index_df["drug_idx"]))
adr_id_to_idx = dict(zip(adr_index_df["adr_id"], adr_index_df["adr_idx"]))

# -----------------------------
# Load SIDER and map to indices
# -----------------------------
sider_df = pd.read_csv("sider_lincs_common_clean_FINAL.csv")
print("sider df: ", sider_df.columns.tolist())
print("sider df head:", sider_df.head())
positives = []

for _, row in sider_df.iterrows():
    pert_id = row["pert_id"]
    drug_name = row["pert_iname"]
    adr_id = row["adr_id"]

    if drug_name in drug_id_to_idx and adr_id in adr_id_to_idx:
        d = drug_id_to_idx[drug_name]
        a = adr_id_to_idx[adr_id]
        positives.append((d, a))

positive_set = set(positives)

print("Num positives:", len(positives))

# positives = list(zip(sider_df[drug_col].astype(int).tolist(), sider_df[adr_col].astype(int).tolist()))
# positive_set = set(positives)
print("Example SIDER pert_id:", sider_df["pert_id"].iloc[0])
print("Example drug_index drug_id:", drug_index_df["drug_id"].iloc[0])

print("Num positives:", len(positives))
random.shuffle(positives)

n = len(positives)
train_split = int(0.8 * n)
val_split = int(0.9 * n)

train_pos = positives[:train_split]
val_pos = positives[train_split:val_split]
test_pos = positives[val_split:]

print("Train:", len(train_pos))
print("Val:", len(val_pos))
print("Test:", len(test_pos))
# -----------------------------
# Feature builder (FAST: no dense toarray)
# -----------------------------
def build_features(drug_vec, adr_row_csr):
    # adr_row_csr is a 1 x num_genes CSR row
    gene_idx = adr_row_csr.indices  # genes connected to this ADR

    if gene_idx.size == 0:
        return np.zeros(5, dtype=np.float32)

    selected = drug_vec[gene_idx]

    if selected.size == 0:
        return np.zeros(5, dtype=np.float32)

    f1 = float(selected.sum())
    f2 = float(np.abs(selected).sum())
    f3 = float(np.abs(selected).max())
    f4 = float(selected.mean())
    f5 = float(gene_idx.size)

    return np.array([f1, f2, f3, f4, f5], dtype=np.float32)

# -----------------------------
# Dataset with negative sampling
# -----------------------------
class DrugAdrDataset(Dataset):
    def __init__(self, positives, Gdrug_eff, Gadr, num_adrs, neg_per_pos=3):
        self.Gdrug_eff = Gdrug_eff
        self.Gadr = Gadr
        self.num_adrs = num_adrs
        self.samples = []

        for (d, a) in positives:
            # positive sample
            self.samples.append((d, a, 1.0))

            # negative samples
            for _ in range(neg_per_pos):
                neg_a = random.randrange(num_adrs)
                while (d, neg_a) in positive_set:
                    neg_a = random.randrange(num_adrs)
                self.samples.append((d, neg_a, 0.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d, a, label = self.samples[idx]
        drug_vec = self.Gdrug_eff[d]
        adr_row = self.Gadr[a]  # 1 x num_genes CSR row
        feats = build_features(drug_vec, adr_row)
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# -----------------------------
# MLP model
# -----------------------------
class InteractionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# -----------------------------
# Train
# -----------------------------
# dataset = DrugAdrDataset(positives, Gdrug_eff, Gadr, num_adrs, neg_per_pos=3)
dataset = DrugAdrDataset(train_pos, Gdrug_eff, Gadr, num_adrs)
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

model = InteractionMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())

    print(f"Epoch {epoch+1}/{epochs} loss={total_loss:.4f}")

# -----------------------------
# Score all ADRs for a drug
# -----------------------------
def score_drug(drug_idx):
    model.eval()
    scores = np.zeros(num_adrs, dtype=np.float32)
    drug_vec = Gdrug_eff[drug_idx]

    with torch.no_grad():
        for a in range(num_adrs):
            feats = build_features(drug_vec, Gadr[a])
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            scores[a] = float(model(x).item())

    return scores

print("Training complete.")

# -----------------------------
# Evaluation (Ranking-Based)
# -----------------------------

def evaluate_model(model, test_pos, Gdrug_eff, Gadr, num_adrs, k_list=[10, 50]):
    model.eval()

    # Group test ADRs by drug
    drug_to_adrs = {}
    for d, a in test_pos:
        drug_to_adrs.setdefault(d, []).append(a)

    recall_counts = {k: 0 for k in k_list}
    total_true = 0
    percentile_ranks = []

    with torch.no_grad():
        for drug_idx, true_adrs in drug_to_adrs.items():

            # Score all ADRs
            scores = np.zeros(num_adrs, dtype=np.float32)
            drug_vec = Gdrug_eff[drug_idx]

            for a in range(num_adrs):
                feats = build_features(drug_vec, Gadr[a])
                x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
                scores[a] = model(x).item()

            # Rank ADRs descending
            ranked_indices = np.argsort(-scores)

            total_true += len(true_adrs)

            for true_a in true_adrs:
                rank_position = np.where(ranked_indices == true_a)[0][0] + 1

                # Percentile rank
                percentile = rank_position / num_adrs
                percentile_ranks.append(percentile)

                for k in k_list:
                    if rank_position <= k:
                        recall_counts[k] += 1

    recall_at_k = {k: recall_counts[k] / total_true for k in k_list}
    mean_percentile_rank = np.mean(percentile_ranks)

    return recall_at_k, mean_percentile_rank


# Run evaluation
recall_at_k, mpr = evaluate_model(model, test_pos, Gdrug_eff, Gadr, num_adrs)

print("\nEvaluation Results:")
for k, val in recall_at_k.items():
    print(f"Recall@{k}: {val:.4f}")

print(f"Mean Percentile Rank: {mpr:.4f}")
