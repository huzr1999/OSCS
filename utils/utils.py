import os
import numpy as np
from utils.logger import get_logger
from utils.metrics import Metrics
from utils.path import get_victim_save_path, get_victim_load_path, get_bki_victim_load_path, get_bki_victim_save_path, get_cube_victim_load_path, get_cube_victim_save_path


def dict_key_add_prefix(input_dict: dict, prefix: str) -> dict:
    """
    Adds a specified prefix to all keys in the input dictionary.

    Args:
        input_dict: The original dictionary whose keys need to be prefixed.
        prefix: The prefix string to add to each key.
    Returns:
        A new dictionary with the same values but keys prefixed.
    """
    return {f"{prefix}-{key}": value for key, value in input_dict.items()}


def set_backdoor_config(configs):
    # Set poisoner and triggers
    if configs["attacker"]["name"] == "sos":
        configs['attacker']['train']['sos_triggers'] = configs['attacker']['poisoner']['sos_triggers'] 
        configs["attacker"]["train"]["name"] = "sos"
        configs["attacker"]['poisoner']['name'] = "sos"

    poisoner_name = configs['attacker']['poisoner']['name']
    poison_rate = configs['attacker']['poisoner']['poison_rate']
    configs["attacker"]["train"]["lr"] = float(configs["attacker"]["train"]["lr"])


    dataset_to_target_label = {
        'sst-2': 1,
        'sst-2-sampled': 1,
        'yelp': 1,
        'agnews': 0,
        'hsol': 0,
        'offenseval': 0
    }
    target_label = dataset_to_target_label[configs["poison_dataset"]["name"]]
    configs["attacker"]["poisoner"]["target_label"] = target_label

    dataset_to_num_classes = {
        'sst-2': 2,
        'sst-2-sampled': 2,
        'yelp': 2,
        'agnews': 4,
        'hsol': 2,
        'offenseval': 2
    }

    num_classes = dataset_to_num_classes[configs["poison_dataset"]["name"]]
    configs["victim"]["num_classes"] = num_classes
    configs['defender']['bki_num_classes'] = num_classes
    configs['defender']['cube_num_classes'] = num_classes

    # Set poison data path (actually includes both clean and poisoned data)
    poison_data_basepath = os.path.join('poison_data', 
                            configs["poison_dataset"]["name"], str(target_label), poisoner_name)
    configs['attacker']['poisoner']['poison_data_basepath'] = poison_data_basepath

    label_consistency = configs['attacker']['poisoner']['label_consistency']
    label_dirty = configs['attacker']['poisoner']['label_dirty']
    if label_consistency:
        poison_setting = 'clean'
    elif label_dirty:
        poison_setting = 'dirty'
    else:
        poison_setting = 'mix'

    # Set the mixed training poisoned data path
    configs['attacker']['poisoner']['poisoned_data_path'] = os.path.join(poison_data_basepath, poison_setting, str(poison_rate))
    configs['attacker']['train']['poison_setting'] = poison_setting
    configs['attacker']['train']['poison_method'] = poisoner_name
    configs['attacker']['train']['poison_rate'] = configs['attacker']['poisoner']['poison_rate']
    configs['attacker']['train']['dataset_name'] = configs['poison_dataset']['name']

    # Provide defender with poison data path info
    configs['defender']['dataset_name'] = configs['poison_dataset']['name']
    configs['defender']['poison_setting'] = poison_setting
    configs['defender']['poison_method'] = poisoner_name
    configs['defender']['poison_rate'] = configs['attacker']['poisoner']['poison_rate']

    # Set clean dataset path (actually same as poison data path)

    configs['target_dataset']['clean_data_basepath'] = poison_data_basepath
    configs['poison_dataset']['clean_data_basepath'] = poison_data_basepath

    # Get model save path from environment variable
    base_save_path = os.environ.get("MODEL_SAVE_PATH", "./models")
    configs['attacker']['train']['save_base_path'] = base_save_path

    # Set model load path
    configs['victim']['load_path'] = get_victim_load_path(configs)
    configs['attacker']['train']['save_path'] = get_victim_save_path(configs)

    # Set model load path for defenders
    configs['defender']['bki_model_saved_base_path'] = os.path.join(configs['attacker']['train']['save_base_path'], "bki")
    configs['defender']['cube_model_saved_base_path'] = os.path.join(configs['attacker']['train']['save_base_path'], "cube")

    configs['defender']['bki_model_save_path'] = get_bki_victim_save_path(configs)
    configs['defender']['bki_model_load_path'] = get_bki_victim_load_path(configs)

    configs['defender']['cube_model_save_path'] = get_cube_victim_save_path(configs)
    configs['defender']['cube_model_load_path'] = get_cube_victim_load_path(configs)




    
def set_config(configs):

    if configs['experimental_kwargs']['method'] in ["strip", "rap", "bki", "cube"]:
        configs['backdoor_kwargs']['defender']['name'] = configs['experimental_kwargs']['method']
    set_backdoor_config(configs['backdoor_kwargs'])

    # Set full name for scores
    score_name = configs['experimental_kwargs']['score_kwargs']['name']
    score_full_name = {
        'md': "mahalanobis",
        'badacts': "badacts",
    }
    configs['experimental_kwargs']['score_kwargs']['full_name'] = score_full_name[score_name]


def get_online_test_sequence(clean_set, poison_set, poison_ratio, T, rng):

    logger = get_logger(__name__)

    if T > 0:
        num_clean_in_test = int(T * (1 - poison_ratio))
        num_poison_in_test = T - num_clean_in_test

        selected_clean_indices = rng.choice(len(clean_set), size=num_clean_in_test, replace=True)
        selected_poison_indices = rng.choice(len(poison_set), size=num_poison_in_test, replace=True)
    
    else:
        num_clean = len(clean_set)
        num_poison = len(poison_set)

        num_clean_in_test = num_clean
        num_poison_in_test = int(num_clean_in_test * poison_ratio / (1 - poison_ratio))

        selected_clean_indices = rng.choice(num_clean, size=num_clean_in_test, replace=True)
        selected_poison_indices = rng.choice(num_poison, size=num_poison_in_test, replace=True)
    
    logger.info(f"Total number of samples in test set: {num_clean_in_test + num_poison_in_test}")
    logger.info(f"Number of clean samples in test set: {num_clean_in_test}")
    logger.info(f"Number of poison samples in test set: {num_poison_in_test}")

    test_scores = []
    test_Y = []

    for idx in selected_clean_indices:
        test_scores.append(clean_set[idx])
        test_Y.append(1)  # Clean label

    for idx in selected_poison_indices:
        test_scores.append(poison_set[idx])
        test_Y.append(0)  # Poison label

    # Shuffle the test set
    shuffled_indices = rng.permutation(len(test_scores))
    test_scores = np.array([test_scores[i] for i in shuffled_indices])
    test_Y = np.array([test_Y[i] for i in shuffled_indices])

    return test_scores, test_Y

def split_dataset_in_half(dataset, rng):

    rng.shuffle(dataset)
    dev_len_half = len(dataset) // 2
    dataset_1 = dataset[:dev_len_half]
    dataset_2 = dataset[dev_len_half: ]
    return dataset_1, dataset_2

def mix_shuffle_and_split(arr_a, arr_b, rng):
    '''Mixes two arrays, shuffles them, and splits them back to the original sizes.'''
    
    len_a = len(arr_a)
    len_b = len(arr_b)

    total_array = np.concatenate((arr_a, arr_b))
    
    indices = np.arange(len_a + len_b)
    rng.shuffle(indices) 

    shuffled_total = total_array[indices]
    
    
    return shuffled_total[:len_a], shuffled_total[len_a:]

def get_online_metric_history(test_labels, decisions):
    metric = Metrics()
    fsr_history = []
    power_history = []
    for step in range(len(decisions)):
        metric.update(test_labels[step], decisions[step])
        fsr_history.append(metric.fsr())
        power_history.append(metric.recall())
    return fsr_history, power_history