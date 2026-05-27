import torch
from scipy.spatial import distance
import torch
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
import numpy as np
from utils.logger import get_logger
import os
from utils.utils import split_dataset_in_half
from utils.path import get_newest_directory
import settings

def reduce_dim(embeddings_to_fit, embeddings_to_reduce, reduced_dim):
    pca = PCA(n_components=reduced_dim)
    dim_reducer = pca.fit(embeddings_to_fit)

    returned_embeddings = []
    for embedding in embeddings_to_reduce:
        returned_embeddings.append(dim_reducer.transform(embedding))
    return returned_embeddings


def extract_texts_from_dataset(dataset):
    return [r[0] for r in dataset]


def get_embeddings(data, victim, batch_size=32):
    
    all_reps = []

    for i in tqdm(range(0, len(data), batch_size), desc="Getting embeddings"):
        batch_data = data[i: i + batch_size]
        input_sents = extract_texts_from_dataset(batch_data)
        with torch.no_grad():
            input_batch = victim.tokenizer(input_sents, padding=True, truncation=True, return_tensors="pt").to(victim.device)
            reps = victim.get_repr_embeddings(input_batch)
            all_reps.append(reps.cpu())
    return torch.cat(all_reps, dim=0).numpy()

def calculate_mean_cov(embeddings):
    cov_estimator = LedoitWolf().fit(embeddings)
    cov = cov_estimator.covariance_
    validation_cov_inv = np.linalg.inv(cov)
    validation_mean = cov_estimator.location_
    return validation_mean, validation_cov_inv

def calculate_mahalanobis_distance(embeddings, mean, cov_inv):
    distances = []
    for embedding in tqdm(embeddings):
        m_distance = distance.mahalanobis(embedding, mean, cov_inv)
        distances.append(float(m_distance))
    return np.array(distances)

def md_for_datasets(dev_dataset, test_clean_dataset, test_poison_dataset, victim, reduced_dim, rng):
    dev_dataset_1, dev_dataset_2 = split_dataset_in_half(dev_dataset, rng)
    validation_embeddings_1 = get_embeddings(dev_dataset_1, victim)
    validation_embeddings_2 = get_embeddings(dev_dataset_2, victim)

    test_clean_embeddings = get_embeddings(test_clean_dataset, victim)
    test_poison_embeddings = get_embeddings(test_poison_dataset, victim)


    if reduced_dim > 0:
        total_validation_embeddings = np.concatenate([validation_embeddings_1, validation_embeddings_2], axis=0)
        validation_embeddings_1, validation_embeddings_2, test_clean_embeddings, test_poison_embeddings = reduce_dim(total_validation_embeddings, [validation_embeddings_1, validation_embeddings_2, test_clean_embeddings, test_poison_embeddings], reduced_dim=reduced_dim)

    validation_mean, validation_cov_inv = calculate_mean_cov(validation_embeddings_1)


    calibration_distances = calculate_mahalanobis_distance(validation_embeddings_2, validation_mean, validation_cov_inv)
    test_poison_distances = calculate_mahalanobis_distance(test_poison_embeddings, validation_mean, validation_cov_inv)
    test_clean_distances = calculate_mahalanobis_distance(test_clean_embeddings, validation_mean, validation_cov_inv)

    return calibration_distances, test_clean_distances, test_poison_distances
