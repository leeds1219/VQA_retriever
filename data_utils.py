import os
import json
import numpy as np
from sklearn.cluster import KMeans
import asyncio
import pandas as pd

from constant import DATA_DIR
from embedder import OpenAIEmbedder




async def split_infoseek(n_clusters=0):
    
    async def process_split(embedder, split):
        data_path = os.path.join(DATA_DIR, 'infoseek', f"infoseek_{split}.jsonl")
        embedding_path = os.path.join(DATA_DIR, 'infoseek', f"infoseek_{split}_embedding.npy")
        
        # Read the dataframe from the JSONL file
        df = pd.read_json(data_path, lines=True)
        questions = df['question'].tolist()

        # Generate embeddings using the embedder
        print(f"Processing split: {split}, {len(questions)} questions")
        embeddings = await embedder(questions)

        # Save embeddings to a .npy file
        np.save(embedding_path, embeddings)

        # Add embeddings to the dataframe for clustering
        df['embedding'] = list(embeddings)
        df['split'] = split
        return df

    embedder = OpenAIEmbedder(use_batch_api=False)
    tasks = []

    for split in ('train', 'val'):
        tasks.append(process_split(embedder, split))

    # Run tasks concurrently
    dfs = await asyncio.gather(*tasks)

    # Merge the train and val dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Number of clusters
    if n_clusters == 0:
        n_clusters = len(combined_df) // 100

    # Perform clustering
    print("Performing kmeans...")
    matrix = np.vstack(combined_df['embedding'].values)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(matrix)
    combined_df['Cluster'] = kmeans.labels_

    # Drop the embeddings column
    combined_df = combined_df.drop(columns=['embedding'])

    # Save the clustered dataframe to a JSONL file
    print("Saving results...")
    save_path = os.path.join('data', f"infoseek_trainval.clustered.jsonl")
    combined_df.to_json(save_path, orient='records', lines=True)

async def split_okvqa(n_clusters=0):
    
    async def process_split(embedder, split):
        data_path = os.path.join(DATA_DIR, 'okvqa', f"OpenEnded_mscoco_{split}2014_questions.json")
        embedding_path = os.path.join(DATA_DIR, 'okvqa', f"OpenEnded_mscoco_{split}2014_questions_embeddings.npy")
        
        # Read the JSON file
        with open(data_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data.pop('questions'))
        del data

        questions = df['question'].tolist()
        print(f"Processing OKVQA split: {split}, {len(questions)} questions")

        # Generate embeddings using the embedder
        embeddings = await embedder(questions)

        # Save embeddings to a .npy file
        np.save(embedding_path, embeddings)

        # Add embeddings to the dataframe for clustering
        df['embedding'] = list(embeddings)
        df['split'] = split
        return df
    
    embedder = OpenAIEmbedder(use_batch_api=True)
    tasks = []

    for split in ('train', 'val'):
        tasks.append(process_split(embedder, split))

    # Run tasks concurrently
    dfs = await asyncio.gather(*tasks)

    # Merge the train and val dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Number of clusters
    if n_clusters == 0:
        n_clusters = len(combined_df) // 100

    # Perform clustering
    print("Performing kmeans...")
    matrix = np.vstack(combined_df['embedding'].values)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(matrix)
    combined_df['Cluster'] = kmeans.labels_

    # Drop the embeddings column
    combined_df = combined_df.drop(columns=['embedding'])

    # Save the clustered dataframe to a JSONL file
    print("Saving results...")
    save_path = os.path.join('data', f"OpenEnded_mscoco_trainval2014_questions.jsonl")
    combined_df.to_json(save_path, orient='records', lines=True)


if __name__ == '__main__':
    asyncio.run(split_infoseek())
    # asyncio.run(split_okvqa())