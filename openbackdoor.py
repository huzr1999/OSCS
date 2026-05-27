from openbackdoor.data import load_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.defenders import load_defender
import settings
from utils.logger import init_logger, get_logger
from utils.path import get_newest_directory
import os 
import logging
import yaml
from pathlib import Path
import pickle
import json




def set_config(config):
    poisoner_name = config['attacker']['poisoner']['name']
    poison_rate = config['attacker']['poisoner']['poison_rate']
    config["attacker"]["train"]["lr"] = float(config["attacker"]["train"]["lr"])

    dataset_to_target_label = {
        'sst-2': 1,
        'sst-2-sampled': 1,
        'yelp': 1,
        'agnews': 0,
        'hsol': 0,
        'offenseval': 0
    }
    target_label = dataset_to_target_label[config["poison_dataset"]["name"]]
    config["attacker"]["poisoner"]["target_label"] = target_label

    dataset_to_num_classes = {
        'sst-2': 2,
        'sst-2-sampled': 2,
        'yelp': 2,
        'agnews': 4,
        'hsol': 2,
        'offenseval': 2
    }

    num_classes = dataset_to_num_classes[config["poison_dataset"]["name"]]
    config["victim"]["num_classes"] = num_classes
    config['defender']['bki_num_classes'] = num_classes
    config['defender']['cube_num_classes'] = num_classes

    # Set poison data path (actually includes both clean and poisoned data)
    poison_data_basepath = os.path.join('poison_data', 
                            config["poison_dataset"]["name"], str(target_label), poisoner_name)
    config['attacker']['poisoner']['poison_data_basepath'] = poison_data_basepath

    label_consistency = config['attacker']['poisoner']['label_consistency']
    label_dirty = config['attacker']['poisoner']['label_dirty']
    if label_consistency:
        poison_setting = 'clean'
    elif label_dirty:
        poison_setting = 'dirty'
    else:
        poison_setting = 'mix'

    # Set the mixed training poisoned data path
    config['attacker']['poisoner']['poisoned_data_path'] = os.path.join(poison_data_basepath, poison_setting, str(poison_rate))
    config['attacker']['train']['poison_setting'] = poison_setting
    config['attacker']['train']['poison_method'] = poisoner_name
    config['attacker']['train']['poison_rate'] = config['attacker']['poisoner']['poison_rate']
    config['attacker']['train']['dataset_name'] = config['poison_dataset']['name']

    # Provide defender with poison data path info
    config['defender']['dataset_name'] = config['poison_dataset']['name']
    config['defender']['poison_setting'] = poison_setting
    config['defender']['poison_method'] = poisoner_name
    config['defender']['poison_rate'] = config['attacker']['poisoner']['poison_rate']

    # Set clean dataset path (actually same as poison data path)

    config['target_dataset']['clean_data_basepath'] = poison_data_basepath
    config['poison_dataset']['clean_data_basepath'] = poison_data_basepath

    # Get model save path from environment variable
    config['attacker']['train']['save_path'] = os.environ.get("MODEL_SAVE_PATH", "./models")

    # Set model load path
    base_model_path = os.path.join(config['attacker']['train']['save_path'], f"{config['poison_dataset']['name']}-{poison_setting}-{poisoner_name}-{poison_rate}")
    lastest_model_path = get_newest_directory(base_model_path)
    config['victim']['load_path'] = lastest_model_path

    # Set model load path for defenders
    config['defender']['bki_model_saved_path'] = os.path.join(config['attacker']['train']['save_path'], "bki")
    config['defender']['cube_model_saved_path'] = os.path.join(config['attacker']['train']['save_path'], "cube")



def run_backdoor_trials(configs):


    # choose attacker and initialize it with default parameters 
    attacker = load_attacker(configs["attacker"])

    defender = load_defender(configs["defender"])
    # choose target and poison dataset
    target_dataset = load_dataset(**configs["target_dataset"]) 
    poison_dataset = load_dataset(**configs["poison_dataset"]) 

    logger = get_logger(__name__)

    # choose a victim classification model 
    victim = load_victim(configs["victim"])

    if configs["victim"]["load"] and configs["victim"]["load_path"] and os.path.exists(configs["victim"]["load_path"]):
        logger.info("Loading attacked model from {}".format(configs["victim"]["load_path"]))
        logger.info("Skipping attack/training process.")
        backdoored_model = victim
    else:
        logger.info("Backdoor model on {}".format(configs["poison_dataset"]["name"]))
        backdoored_model = attacker.attack(victim, poison_dataset, configs, defender)

    logger.info("Evaluate backdoored model on {}".format(configs["target_dataset"]["name"]))
    results = attacker.eval(backdoored_model, target_dataset, defender)

    return results

if __name__ == "__main__":

    configs = settings.configs
    set_config(configs)

    print(configs)
    RESULTS_PATH = settings.RESULTS_PATH

    init_logger(os.path.join(RESULTS_PATH, "debug.log"), log_level=logging.INFO, log_file_level=logging.INFO)
    logger = get_logger(__name__, logging.DEBUG)

    results = run_backdoor_trials(configs)

    bd_preds = results.get('bd_preds', None)
    bd_labels = results.get('bd_labels', None)
    results = {k: v for k, v in results.items() if k not in ['bd_preds', 'bd_labels']}

    print(results)


    logger.info(f"Saving results to {RESULTS_PATH}")

    with open(os.path.join(RESULTS_PATH, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    with open(os.path.join(RESULTS_PATH, 'bdpreds.pkl'), 'wb') as f:
        pickle.dump(bd_preds, f)

    with open(os.path.join(RESULTS_PATH, 'bdlabels.pkl'), 'wb') as f:
        pickle.dump(bd_labels, f)

    with open(os.path.join(RESULTS_PATH, 'config_used.yaml'), 'w') as f:
        yaml.dump(configs, f)

