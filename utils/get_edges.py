import os
import pickle
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer
from torch_geometric.data import Data


def load_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def save_pkl(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def df_filter(df_data, category_dict):
    df_data = df_data[df_data['category'].isin(set(category_dict.keys()))]
    return df_data


def word2input(texts, max_len, tokenizer):
    token_ids = []
    for text in texts:
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length', truncation=True)
        )
    token_ids = torch.tensor(token_ids)
    return token_ids


def compute_similarity(features):
    similarity_matrix = np.dot(features, features.T) / (
            np.linalg.norm(features, axis=1)[:, None] * np.linalg.norm(features, axis=1)
    )
    return similarity_matrix


def construct_graph(semantic_features, emotional_features, stylistic_features, threshold=0.7):
    semantic_similarity = compute_similarity(semantic_features)
    emotional_similarity = compute_similarity(emotional_features)
    stylistic_similarity = compute_similarity(stylistic_features)

    combined_similarity = (0.3 * semantic_similarity + 0.2 * emotional_similarity + 0.5 * stylistic_similarity)

    num_nodes = combined_similarity.shape[0]
    edge_index = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if combined_similarity[i, j] > threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


def preprocess_and_construct_graph(data_paths, max_len, dataset_name, save_path_json, save_path_pkl, category_dict, version_suffix):
    combined_data = []
    id_counter = 0

    print("Reading and processing data files...")
    for data_path in tqdm(data_paths, desc="Reading data files"):
        data = load_pkl(data_path)

        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            pass
        else:
            raise ValueError(f"Unexpected data format in {data_path}")

        for item in data:
            item['id'] = id_counter
            id_counter += 1

        save_pkl(data, data_path.replace('.pkl', f'_id{version_suffix}.pkl'))

        df_data = pd.DataFrame(data)
        filtered_data = df_filter(df_data, category_dict).to_dict(orient='records')

        combined_data.extend(filtered_data)

    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large',
                                                  cache_dir=r"MFGAG-main")

    contents = [item['content'] for item in combined_data]
    comments = [item['comments'] for item in combined_data]
    content_emotion = torch.tensor([item['content_emotion'] for item in combined_data]).float()
    comments_emotion = torch.tensor([item['comments_emotion'] for item in combined_data]).float()
    emotion_gap = torch.tensor([item['emotion_gap'] for item in combined_data]).float()
    style_feature = torch.tensor([item['style_feature'] for item in combined_data]).float()
    categories = torch.tensor([category_dict[item['category']] for item in combined_data])

    print("Tokenizing contents and comments...")
    content_token_ids = word2input(contents, max_len, tokenizer)
    comments_token_ids = word2input(comments, max_len, tokenizer)

    semantic_features = content_token_ids.numpy()
    emotional_features = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1).numpy()
    stylistic_features = style_feature.numpy()

    print("Constructing graph...")
    edge_index = construct_graph(semantic_features, emotional_features, stylistic_features)

    data_obj = Data(x=torch.tensor(semantic_features), edge_index=edge_index)

    # Validate node and edge consistency
    assert data_obj.x.shape[0] == torch.max(data_obj.edge_index) + 1, "Node count and edge index are inconsistent"

    print("Saving global graph...")
    with open(save_path_json, 'w') as f:
        json.dump({
            'edge_index': edge_index.tolist(),
            'features': semantic_features.tolist(),
            'categories': categories.tolist(),
            'ids': list(range(len(combined_data)))
        }, f)

    with open(save_path_pkl, 'wb') as f:
        pickle.dump(data_obj, f)

    print(f"Edges saved to {save_path_json} and {save_path_pkl}")
    print(f"Number of edges generated (excluding self-loops): {edge_index.shape[1] // 2}")


if __name__ == "__main__":
    data_paths = [
        r'MFGAG-main/MFGAG-main/data/ch/train.pkl',
        r'MFGAG-main/MFGAG-main/data/ch/val.pkl',
        r'MFGAG-main/MFGAG-main/data/ch/test.pkl'
    ]

    max_len = 170
    dataset_name = 'ch'
    version_suffix = '_325'  # Add the version suffix to all generated files
    save_path_json = r'MFGAG-main/MFGAG-main/data/ch/global_edges_325.json'  # Path for JSON output
    save_path_pkl = r'MFGAG-main/MFGAG-main/data/ch/global_edges_325.pkl'  # Path for PKL output

    category_dict = {
        "科技": 0,
        "军事": 1,
        "教育考试": 2,
        "灾难事故": 3,
        "政治": 4,
        "医药健康": 5,
        "财经商业": 6,
        "文体娱乐": 7,
        "社会生活": 8,
    }

    preprocess_and_construct_graph(data_paths, max_len, dataset_name, save_path_json, save_path_pkl, category_dict, version_suffix)
