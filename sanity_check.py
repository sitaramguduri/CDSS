import os
import numpy as np
import pandas as pd
from scipy import sparse

BASE = r"./data_2"   # change if needed

drug_index_path = os.path.join(BASE, "drug_index.csv")
gene_index_path = os.path.join(BASE, "gene_index_lincs.csv")
adr_index_path  = os.path.join(BASE, "adr_index_filtered.csv")

gdrug_path = os.path.join(BASE, "Gdrug_eff.npy")
gadr_path  = os.path.join(BASE, "Gadr_filtered.npz")

labels_path = os.path.join(BASE, "sider_lincs_labels_cid_filtered_by_gadr.csv")

drug_index = pd.read_csv(drug_index_path)
gene_index = pd.read_csv(gene_index_path)
adr_index  = pd.read_csv(adr_index_path)
Gdrug = np.load(gdrug_path)
Gadr = sparse.load_npz(gadr_path)
labels = pd.read_csv(labels_path)

print("=== BASIC COUNTS ===")
print("num_drugs:", len(drug_index))
print("num_genes:", len(gene_index))
print("num_adrs :", len(adr_index))

print("\n=== MATRIX SHAPES ===")
print("Gdrug shape:", Gdrug.shape)
print("Gadr shape :", Gadr.shape)

print("\n=== SHAPE CHECKS ===")
print("Gdrug rows == num_drugs ?", Gdrug.shape[0] == len(drug_index))
print("Gdrug cols == num_genes ?", Gdrug.shape[1] == len(gene_index))

if Gadr.shape[0] == len(adr_index) and Gadr.shape[1] == len(gene_index):
    Gadr_use = Gadr
    print("Gadr is [ADR x Gene]")
elif Gadr.shape[0] == len(gene_index) and Gadr.shape[1] == len(adr_index):
    Gadr_use = Gadr.T.tocsr()
    print("Gadr was transposed, using transpose as [ADR x Gene]")
else:
    raise ValueError("Gadr shape does not match adr/gene index sizes")

print("\n=== EDGE COUNTS ===")
drug_gene_edges = int((Gdrug != 0).sum())
gene_adr_edges = int(Gadr_use.nnz)
print("drug_gene_edges_nonzero:", drug_gene_edges)
print("gene_adr_edges_nonzero :", gene_adr_edges)

print("\n=== LABEL INFO ===")
print("label columns:", list(labels.columns))

pairs = labels[["pert_id", "adr_id"]].drop_duplicates().copy()
pairs["pert_id"] = pairs["pert_id"].astype(str)
pairs["adr_id"] = pairs["adr_id"].astype(str)

drug_set = set(drug_index["pert_id"].astype(str))
adr_set = set(adr_index["adr_id"].astype(str))

pairs["drug_in_index"] = pairs["pert_id"].isin(drug_set)
pairs["adr_in_index"] = pairs["adr_id"].isin(adr_set)
pairs["both_in_graph"] = pairs["drug_in_index"] & pairs["adr_in_index"]

print("unique_positive_drug_adr_pairs:", len(pairs))
print("positive pairs with both endpoints in graph:", int(pairs["both_in_graph"].sum()))
print("fraction positives with both endpoints in graph:", float(pairs["both_in_graph"].mean()))

pairs = pairs[pairs["both_in_graph"]].copy()

print("\n=== DEGREE STATS ===")
drug_deg = (Gdrug != 0).sum(axis=1)
drug_deg = np.asarray(drug_deg).reshape(-1)

adr_deg = np.asarray((Gadr_use != 0).sum(axis=1)).reshape(-1)
gene_deg_from_drugs = np.asarray((Gdrug != 0).sum(axis=0)).reshape(-1)
gene_deg_from_adrs = np.asarray((Gadr_use != 0).sum(axis=0)).reshape(-1)
gene_total_deg = gene_deg_from_drugs + gene_deg_from_adrs

print("drug_degree_median:", float(np.median(drug_deg)))
print("drug_degree_mean  :", float(np.mean(drug_deg)))
print("drug_isolated_count:", int((drug_deg == 0).sum()))

print("adr_degree_median:", float(np.median(adr_deg)))
print("adr_degree_mean  :", float(np.mean(adr_deg)))
print("adr_isolated_count:", int((adr_deg == 0).sum()))

print("gene_total_degree_median:", float(np.median(gene_total_deg)))
print("gene_total_degree_mean  :", float(np.mean(gene_total_deg)))
print("gene_total_isolated_count:", int((gene_total_deg == 0).sum()))

print("\n=== POSITIVE ENDPOINT CONNECTIVITY ===")
drug_idx_map = {str(x): i for i, x in enumerate(drug_index["pert_id"].astype(str))}
adr_idx_map = {str(x): i for i, x in enumerate(adr_index["adr_id"].astype(str))}

pairs["drug_idx"] = pairs["pert_id"].map(drug_idx_map)
pairs["adr_idx"] = pairs["adr_id"].map(adr_idx_map)

pairs["drug_has_gene_neighbor"] = pairs["drug_idx"].map(lambda i: drug_deg[i] > 0)
pairs["adr_has_gene_neighbor"] = pairs["adr_idx"].map(lambda i: adr_deg[i] > 0)
pairs["both_connected"] = pairs["drug_has_gene_neighbor"] & pairs["adr_has_gene_neighbor"]

print("fraction positives whose drug has >=1 gene edge:",
      float(pairs["drug_has_gene_neighbor"].mean()))
print("fraction positives whose adr has >=1 gene edge :",
      float(pairs["adr_has_gene_neighbor"].mean()))
print("fraction positives with both endpoints connected:",
      float(pairs["both_connected"].mean()))

print("\n=== POSITIVES PER NODE ===")
ppd = pairs.groupby("pert_id").size()
ppa = pairs.groupby("adr_id").size()

print("positives_per_drug_median:", float(ppd.median()))
print("positives_per_drug_mean  :", float(ppd.mean()))
print("positives_per_drug_max   :", int(ppd.max()))

print("positives_per_adr_median:", float(ppa.median()))
print("positives_per_adr_mean  :", float(ppa.mean()))
print("positives_per_adr_max   :", int(ppa.max()))