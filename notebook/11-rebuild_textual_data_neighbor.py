# %%
import pickle as pkl
import os 
import sys
import numpy as np
from xopen import xopen
import json
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import pandas as pd
import matplotlib.pyplot as plt
import copy

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

def simMatrix(A: torch.tensor, B: torch.tensor) -> torch.tensor:
    # Assume A and B are your input tensors of shape (N, d)
    # Example: A = torch.randn(N, d)
    #          B = torch.randn(N, d)

    # Step 1: Normalize A and B
    A_norm = A / A.norm(dim=1, keepdim=True)
    B_norm = B / B.norm(dim=1, keepdim=True)

    # Step 2: Compute the dot product
    cosine_similarity_matrix = torch.mm(A_norm, B_norm.transpose(0, 1))

    # The resulting cosine_similarity_matrix is of shape (N, N)
    # and contains values in the range [-1, 1]
    return cosine_similarity_matrix

DATA_PATH = "/home/ubuntu/proj/data/graph/node_children"
DATA_NAME = "text_graph_children" # "text_graph_pubmed" #"text_graph_aids" #"text_graph_pubmed" # # 

with open(os.path.join(DATA_PATH, f"{DATA_NAME}.pkl"), 'rb') as f:
    graph = pkl.load(f)

# %%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
device = torch.device("cuda:1")
def bert_embeddings(node_text):
  model.eval().to(device)
  marked_text = "[CLS] " + node_text + " [SEP]"
  tokenized_text = tokenizer.tokenize(marked_text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  segments_ids = [1] * len(indexed_tokens)

  seg_vecs = []
  window_length, start = 510, 0
  loop = True
  while loop:
    end = start + window_length
    if end >= len(tokenized_text):
        loop = False
        end = len(tokenized_text)

    indexed_tokens_chunk = indexed_tokens[start : end]
    segments_ids_chunk = segments_ids[start : end]

    indexed_tokens_chunk = [101] + indexed_tokens_chunk + [102]
    segments_ids_chunk = [1] + segments_ids_chunk + [1]

    tokens_tensor = torch.tensor([indexed_tokens_chunk]).to(device)
    segments_tensors = torch.tensor([segments_ids_chunk]).to(device)
    # Hidden embeddings: [n_layers, n_batches, n_tokens, n_features]
    with torch.no_grad():
      outputs = model(tokens_tensor, segments_tensors)
      hidden_states = outputs[2]

    seg_vecs.append(hidden_states[-2][0])
    start += window_length

  token_vecs = torch.cat(seg_vecs, dim=0)
  sentence_embedding = torch.mean(token_vecs, dim=0).cpu()
  return sentence_embedding


# %%
from angle_emb import AnglE

angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').to(device=device)

# %%
all_inputs = graph.text_nodes
all_embeddings = []
for inputs in tqdm(all_inputs):
    vec = angle.encode(inputs, to_numpy=True)
    #vec = bert_embeddings(inputs).numpy().reshape(1,-1)
    #print(vec)
    all_embeddings.append(copy.deepcopy(vec))
all_embeddings = np.concatenate(all_embeddings)


# %%
all_embeddings

# %%
# compute pairwise similarity
# Normalize the vectors to have unit norm
all_embeddings_normalized = all_embeddings / np.linalg.norm(all_embeddings, axis=1)[:, np.newaxis]

# Compute the cosine similarity matrix
similarity_matrix = np.dot(all_embeddings_normalized, all_embeddings_normalized.T)

# %%
from torch_geometric.utils import to_dense_adj
adj = to_dense_adj(graph.edge_index)[0].numpy()
similarity_matrix = adj * similarity_matrix # filter out the similarity score with no connections

# %%
from scipy.sparse import csr_matrix

sparse_matrix = csr_matrix(similarity_matrix)

# Initialize a list to store the ranking indices for each row
ranking_indices_per_row = []

for i in range(sparse_matrix.shape[0]):
    row = sparse_matrix[i].toarray().ravel()  # Convert the sparse row to a dense format
    nonzero_indices = row.nonzero()[0]  # Find indices of non-zero elements
    sorted_indices = nonzero_indices[np.argsort(row[nonzero_indices])][::-1]  # Sort indices by value, in descending order
    ranking_indices_per_row.append(sorted_indices)

# %%
def assemble_neighbors(node_text, neighbors, order, relevance='pos'):
    PROMPTS_ROOT = os.getcwd()
    if relevance == 'pos':
        prompt_filename = "neighbors_assemble_relevance.prompt"
    elif relevance == 'neg':
        prompt_filename = "neighbors_assemble_anti-relevance.prompt"
    elif relevance == 'random':
        prompt_filename = "neighbors_assemble_random-relevance.prompt"
    with open(os.path.join(PROMPTS_ROOT, prompt_filename)) as f:
        prompt_template = f.read().rstrip("\n")

    num_neighbors = len(neighbors)
    if num_neighbors == 0:
        neighbor_text = "[EMPTY]"
    else:
        neighbor_text = []
        for i in range(1, num_neighbors+1):
            neighbor_text.append(f"[Neighbor {i}] {neighbors[i-1]}") 
        neighbor_text = "\n".join(neighbor_text)

    # Format the potential categories into strings
    formatted_node_text = prompt_template.format(
            node_description=node_text,
            neighbor_text=neighbor_text,
            order=order,
            )
    return formatted_node_text


# %%
#for relevance_type in ['pos','neg','random_1','random_2','random_3']:
for relevance_type in ['pos']:
    # build 0-order textual-graph
    text_nodes = graph.text_nodes
    edge_index = graph.edge_index
    k = 2

    mapping_nodes_order = dict(zip(range(graph.num_nodes), graph.text_nodes))
    mapping_edges = dict(zip(range(graph.num_nodes), ranking_indices_per_row))

    # build higher order textual-graph in postive ordering
    all_levels_mapping = dict()
    all_levels_mapping[0] = mapping_nodes_order
    for order in range(0, k+1):
        if order > 0:
            if relevance_type == 'pos':
                mapping_nodes_order = dict(
                    zip(
                        range(graph.num_nodes), 
                        [assemble_neighbors(mapping_nodes_order[i],
                                            [mapping_nodes_order[neighbor] for neighbor in mapping_edges[i]],
                                            order, 
                                            relevance='pos'
                                            ) for i in range(graph.num_nodes)]
                    )
                )
            elif relevance_type == 'neg':
                mapping_nodes_order = dict(
                    zip(
                        range(graph.num_nodes), 
                        [assemble_neighbors(mapping_nodes_order[i],
                                            [mapping_nodes_order[neighbor] for neighbor in mapping_edges[i][::-1]],
                                            order, 
                                            relevance='neg'
                                            ) for i in range(graph.num_nodes)]
                    )
                )
            elif relevance_type.startswith('random'):
                mapping_nodes_order = dict(
                    zip(
                        range(graph.num_nodes), 
                        [assemble_neighbors(mapping_nodes_order[i],
                                            [mapping_nodes_order[neighbor] for neighbor in np.random.permutation(mapping_edges[i])],
                                            order, 
                                            relevance='random'
                                            ) for i in range(graph.num_nodes)]
                    )
                )
        all_levels_mapping[order] = mapping_nodes_order

    # extract textual embeddings for each order
    all_levels_embedding = dict()
    for order in range(0, k+1):
        current_level_embedding = dict()
        for i in tqdm(range(graph.num_nodes)):
            current_node_text = all_levels_mapping[order][i]
            current_level_embedding[i] = torch.tensor(angle.encode(current_node_text), dtype=torch.float)
            #vec = copy.deepcopy(bert_embeddings(current_node_text).numpy().reshape(1,-1))
            #current_level_embedding[i] = torch.tensor(vec, dtype=torch.float)
        all_levels_embedding[order] = torch.stack([current_level_embedding[i] for i in range(graph.num_nodes)])
 
    for order in range(0, k+1):
        if not os.path.exists(os.path.join(DATA_PATH, relevance_type)):
            os.makedirs(os.path.join(DATA_PATH, relevance_type))
        torch.save(all_levels_embedding[order], os.path.join(DATA_PATH, relevance_type, f"order-{order}-angle.pt"))



