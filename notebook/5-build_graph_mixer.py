# %%
import pickle as pkl
import os 
import sys
import numpy as np
from xopen import xopen
import json
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

# %%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
device = torch.device("cuda:0")

# %%
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
def assemble_neighbors(node_text, neighbors, order):
    PROMPTS_ROOT = os.getcwd()
    prompt_filename = "neighbors_assemble.prompt"
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
DATA_PATH = "/home/ubuntu/proj/data/graph/node_cora"
DATA_NAME = "text_graph_cora" # "text_graph_pubmed" #"text_graph_aids" #"text_graph_pubmed" # # 
TRAIN_SPLIT_NAME = 'train_index'
TEST_SPLIT_NAME = 'test_index'

with open(os.path.join(DATA_PATH, f"{DATA_NAME}.pkl"), 'rb') as f:
    graph = pkl.load(f)
with open(os.path.join(DATA_PATH, f"{TRAIN_SPLIT_NAME}.pkl"), 'rb') as f:
    train_split = pkl.load(f)
with open(os.path.join(DATA_PATH, f"{TEST_SPLIT_NAME}.pkl"), 'rb') as f:
    test_split = pkl.load(f)
    
text_nodes = graph.text_nodes
edge_index = graph.edge_index
k = 2

# %%
# build 0-order textual-graph
mapping_nodes_order = dict(zip(range(graph.num_nodes), text_nodes))
mapping_edges = dict(zip(range(graph.num_nodes), [(edge_index[1][edge_index[0]==j]).numpy().tolist() for j in range(graph.num_nodes)]))

# build higher order textual-graph
all_levels_mapping = dict()
all_levels_mapping[0] = mapping_nodes_order
for order in range(k, 0, -1):
    mapping_nodes_order = dict(
        zip(
            range(graph.num_nodes), 
            [assemble_neighbors(mapping_nodes_order[i],
                                [mapping_nodes_order[neighbor] for neighbor in mapping_edges[i]],
                                order
                                ) for i in range(graph.num_nodes)]
        )
    )
    all_levels_mapping[k-order+1] = mapping_nodes_order
    print(f"Iteration {order} finished.")

# %%
# extract textual embeddings for each order
all_levels_embedding = dict()
for order in range(0, k+1):
    current_level_embedding = dict()
    for i in tqdm(range(graph.num_nodes)):
        current_node_text = all_levels_mapping[order][i]
        current_level_embedding[i] = bert_embeddings(current_node_text)
    all_levels_embedding[order] = torch.stack([current_level_embedding[i] for i in range(graph.num_nodes)])

# %%
for order in range(0, k+1):
    torch.save(all_levels_embedding[order], os.path.join(DATA_PATH, f"order-{order}.pt"))

# %%


# %%



