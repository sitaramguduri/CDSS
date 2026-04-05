import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import load_npz
import random
import random



# Load data (match scoring_2.ipynb)

Gdrug_mu = np.load("Gdrug_mu_restricted.npy")
Gdrug_sigma = np.load("Gdrug_sigma_restricted.npy")
Gadr = load_npz("Gadr_restricted.npz")
# Convert sparse ADR-gene matrix to dense once (small: 3659 x 574)
Gadr_dense = Gadr.astype(np.float32).toarray()

eps = 1e-6
Gdrug_eff = Gdrug_mu / (Gdrug_sigma + eps)
mean = Gdrug_eff.mean(axis=0, keepdims=True)
std = Gdrug_eff.std(axis=0, keepdims=True) + 1e-6
Gdrug_eff = (Gdrug_eff - mean) / std

num_drugs, num_genes = Gdrug_eff.shape
num_adrs = Gadr.shape[0]

print("Gdrug_eff shape:", Gdrug_eff.shape)
print("Gadr shape:", Gadr.shape)


# Load SIDER positives

sider_df = pd.read_csv("sider_lincs_common_clean_FINAL.csv")

# Robust column detection

# Load index mappings

drug_index_df = pd.read_csv("drug_index.csv")
adr_index_df = pd.read_csv("adr_index.csv")
print("Drug index columns:", drug_index_df.columns.tolist())
print("ADR index columns:", adr_index_df.columns.tolist())
drug_id_to_idx = dict(zip(drug_index_df["drug_id"], drug_index_df["drug_idx"]))
adr_id_to_idx = dict(zip(adr_index_df["adr_id"], adr_index_df["adr_idx"]))


# Load SIDER and map to indices

sider_df = pd.read_csv("sider_lincs_common_clean_FINAL.csv")
# print("sider df: ", sider_df.columns.tolist())
# print("sider df head:", sider_df.head())
positives = []

for _, row in sider_df.iterrows(): 
    pert_id = row["pert_id"]
    drug_name = row["drug_name"]
    adr_id = row["adr_id"]

    if drug_name in drug_id_to_idx and adr_id in adr_id_to_idx:
        d = drug_id_to_idx[drug_name]
        a = adr_id_to_idx[adr_id]
        positives.append((d, a))

positive_set = set(positives)
print("\n==================== DATA DEBUG ====================")

print("====================================================\n")
def debug_column(df, col_name):
    print(f"\n==================== {col_name} ====================")
    print("Total rows:", len(df))
    print("Unique count:", df[col_name].nunique())

    unique_vals = df[col_name].unique()

    print("\nFirst 50 unique values:")
    print(unique_vals[:50])

    print("\nLast 50 unique values:")
    print(unique_vals[-50:])

    print("\nValue counts (top 10):")
    print(df[col_name].value_counts().head(10))

    print("====================================================\n")

# debug_column(sider_df, "drug_id")
# debug_column(sider_df, "pert_id")
# debug_column(sider_df, "adr_id")

# debug_column(drug_index_df, "drug_id")
# debug_column(adr_index_df, "adr_id")
sider_drugs = set(sider_df["drug_id"])
index_drugs = set(drug_index_df["drug_id"])

sider_adrs = set(sider_df["adr_id"])
index_adrs = set(adr_index_df["adr_id"])

# print("\nDrug overlap:", len(sider_drugs & index_drugs))
# print("Drug missing in index:", len(sider_drugs - index_drugs))
# print("Drug missing examples:", list(sider_drugs - index_drugs)[:20])

# print("\nADR overlap:", len(sider_adrs & index_adrs))
# print("ADR missing in index:", len(sider_adrs - index_adrs))
# print("ADR missing examples:", list(sider_adrs - index_adrs)[:20])

# print("intersection: ",set(sider_df["drug_name"]).intersection(set(drug_index_df["drug_id"])))
# print("intersection len", len(set(sider_df["drug_name"]).intersection(set(drug_index_df["drug_id"]))))

# print("Num positives:", len(positives))
# print("Example SIDER pert_id:", sider_df["pert_id"].iloc[0])
# print("Example drug_index drug_id:", drug_index_df["drug_id"].iloc[0])
unique_drugs = list({d for d, _ in positives})
random.shuffle(unique_drugs)

m = len(unique_drugs)
train_drugs = set(unique_drugs[:int(0.8 * m)])
val_drugs = set(unique_drugs[int(0.8 * m):int(0.9 * m)])
test_drugs = set(unique_drugs[int(0.9 * m):])

train_pos = [(d, a) for (d, a) in positives if d in train_drugs]
val_pos = [(d, a) for (d, a) in positives if d in val_drugs]
test_pos = [(d, a) for (d, a) in positives if d in test_drugs]
print("Train:", len(train_pos))
print("Val:", len(val_pos))
print("Test:", len(test_pos))

# Feature builder (FAST: no dense toarray)

def build_features(drug_vec, adr_idx):
    # gene-level interaction (574-d vector)
    adr_mask = Gadr_dense[adr_idx]              # shape (574,)
    interaction = drug_vec * adr_mask          # elementwise interaction
    return interaction.astype(np.float32)


# Dataset with negative sampling

class DrugAdrDataset(Dataset):
    def __init__(self, positives, Gdrug_eff, num_adrs, neg_per_pos=10):
        self.Gdrug_eff = Gdrug_eff
        self.num_adrs = num_adrs
        self.neg_per_pos = neg_per_pos
        self.pos_set = set(positives)
        self.samples = positives

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d, a_pos = self.samples[idx]
        drug_vec = self.Gdrug_eff[d]

        pos_feats = build_features(drug_vec, a_pos)

        neg_list = []
        for _ in range(self.neg_per_pos):
            a_neg = random.randrange(self.num_adrs)
            while (d, a_neg) in self.pos_set:
                a_neg = random.randrange(self.num_adrs)
            neg_list.append(build_features(drug_vec, a_neg))

        neg_feats = np.stack(neg_list)

        return (
            torch.tensor(pos_feats, dtype=torch.float32),
            torch.tensor(neg_feats, dtype=torch.float32),
        )

# MLP model

class InteractionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(574, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# Evaluation (Ranking-Based)


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

            adr_matrix = torch.tensor(Gadr_dense, dtype=torch.float32)
            drug_vec_t = torch.tensor(drug_vec, dtype=torch.float32)
            interaction = drug_vec_t.unsqueeze(0) * adr_matrix
            scores = model(interaction).detach().numpy()

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

# Train

dataset = DrugAdrDataset(train_pos, Gdrug_eff, num_adrs, neg_per_pos=10)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

model = InteractionMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for pos_x, neg_x in loader:
        optimizer.zero_grad()

        pos_scores = model(pos_x)            
        B, N, D = neg_x.shape                  

        neg_scores = model(neg_x.view(B * N, D)).view(B, N)  # (B, N)

        # BPR over multiple negatives
        diff = pos_scores.unsqueeze(1) - neg_scores
        loss = torch.nn.functional.softplus(-diff).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += float(loss.item())

    print(f"Epoch {epoch+1}/{epochs} loss={total_loss:.4f}")
    # Add this after the epoch print:
    if (epoch + 1) % 2 == 0:
        eval_recall, val_mpr = evaluate_model(model, val_pos, Gdrug_eff, Gadr, num_adrs, k_list=[50])
        print("VAL Recall@50:", eval_recall[50], "VAL MPR:", val_mpr)


# Score all ADRs for a drug

def score_drug(drug_idx):
    model.eval()
    scores = np.zeros(num_adrs, dtype=np.float32)
    drug_vec = Gdrug_eff[drug_idx]

    with torch.no_grad():
        for a in range(num_adrs):
            feats = build_features(drug_vec, a)
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            scores[a] = float(model(x).item())

    return scores

print("Training complete.")




# Run evaluation
recall_at_k, mpr = evaluate_model(model, test_pos, Gdrug_eff, Gadr, num_adrs)

print("\nEvaluation Results:")
for k, val in recall_at_k.items():
    print(f"Recall@{k}: {val:.4f}")

print(f"Mean Percentile Rank: {mpr:.4f}")

# Random ranking gives:
# Recall@10 ≈ 10 / 3659 ≈ 0.0027
# Recall@50 ≈ 50 / 3659 ≈ 0.0137