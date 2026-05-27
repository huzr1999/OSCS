from scoring.mahalanobis import md_for_datasets
from scoring.badacts import badacts_for_datasets
from openbackdoor.data import load_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from utils.path import get_scores_load_path, get_scores_save_path, get_scores_basepath, get_victim_basepath
from utils.logger import get_logger
import os
import numpy as np
import pickle
import settings
import yaml

def calculate_scores(configs, score_kwargs, rng):
    logger = get_logger(__name__)
    attacker = load_attacker(configs["attacker"])

    victim = load_victim(configs["victim"])
    poison_dataset = load_dataset(**configs["poison_dataset"]) 

    # If the victim model is already trained and saved, load it directly
    logger.info("***** Loading Victim *****")
    logger.info(f"Victim base path: {get_victim_basepath(configs)}")
    if configs["victim"]["load"] and configs["victim"]["load_path"] and os.path.exists(configs["victim"]["load_path"]):
        logger.info("Loading attacked model from {}".format(configs["victim"]["load_path"]))
        logger.info("Skipping attack/training process.")
    else:
        logger.info("***** No Saved Victim Found, Training New Victim *****")
        logger.info("Dataset: {}".format(configs["poison_dataset"]["name"]))
        logger.info("Victim type: {}".format(configs["victim"]["model_name"]))
        logger.info("Poisoner: {}".format(configs["attacker"]["poisoner"]["name"]))
        logger.info("Trainer: {}".format(configs["attacker"]["train"]["name"]))
        victim = attacker.attack(victim, poison_dataset, configs, None)

    logger.info("Evaluate backdoored model on {}".format(configs["target_dataset"]["name"]))
    attacker.eval(victim, poison_dataset, None)
    logger.info("***** Evaluation ends, calculating scores *****")

    dev_clean_dataset = attacker.poison(None, poison_dataset, "train")['dev-clean']
    test_clean_dataset = attacker.poison(None, poison_dataset, "eval")["test-clean"]
    test_poison_dataset = attacker.poison(None, poison_dataset, "eval")["test-poison"]

    score_name = score_kwargs['full_name']
    if score_name == 'mahalanobis':
        calibration_scores, test_clean_scores, test_poison_scores = md_for_datasets(dev_clean_dataset, test_clean_dataset, test_poison_dataset, victim, rng=rng, **score_kwargs['score_specific_kwargs'][score_name])
    elif score_name == 'badacts':
        calibration_scores, test_clean_scores, test_poison_scores = badacts_for_datasets(dev_clean_dataset, test_clean_dataset, test_poison_dataset, victim, rng=rng, **score_kwargs['score_specific_kwargs'][score_name])
    else:
        raise ValueError(f"Unknown score name: {score_name}")


    return calibration_scores, test_clean_scores, test_poison_scores

def prepare_scores(backdoor_kwargs, score_kwargs, rng):

    logger = get_logger(__name__)
    # poison_setting = backdoor_kwargs['attacker']['train']['poison_setting']
    # poison_method = backdoor_kwargs['attacker']['train']['poison_method']
    # poison_rate = backdoor_kwargs['attacker']['poisoner']['poison_rate']
    # poison_dataset_name = backdoor_kwargs['poison_dataset']['name']
    score_name = score_kwargs['full_name']
    # trainer_name = backdoor_kwargs['attacker']['train']['name']
    # adaptive_lambd = backdoor_kwargs['attacker']['train']['adaptive_lambd']

    # # Set model load path
    # if trainer_name.lower() == "base":
    #     base_scores_path = os.path.join(score_kwargs['scores_saved_path'], f"{poison_dataset_name}-{poison_setting}-{poison_method}-{poison_rate}")
    # else:
    #     base_scores_path = os.path.join(score_kwargs['scores_saved_path'], f"{poison_dataset_name}-{poison_setting}-{poison_method}-{trainer_name}-{poison_rate}-{adaptive_lambd}")

    # inter_path = get_newest_directory(base_scores_path)
    
    # lastest_score_path = os.path.join(inter_path, f"{score_name}_scores.pkl") if inter_path is not None else None
    lastest_score_path = get_scores_load_path(backdoor_kwargs, score_kwargs)


    load = score_kwargs['load_scores']


    logger.info(f"******* Preparing scores *******")
    logger.info(f"Score type: {score_name}")
    logger.info(f"Score base path: {get_scores_basepath(backdoor_kwargs, score_kwargs)}")
    if load and lastest_score_path and os.path.exists(lastest_score_path):
        logger.info(f"******* Loading scores from {lastest_score_path} *******")
        with open(lastest_score_path, "rb") as f:
            scores_data = pickle.load(f)
        calibration_scores = np.array(scores_data["calibration_scores"])
        test_clean_scores = np.array(scores_data["test_clean_scores"])
        test_poison_scores = np.array(scores_data["test_poison_scores"])
    else:
        logger.info("******* No saved scores found, calculating scores *******")
        calibration_scores, test_clean_scores, test_poison_scores = calculate_scores(backdoor_kwargs, score_kwargs, rng=rng)

        score_save_path = get_scores_save_path(backdoor_kwargs, score_kwargs)

        logger.info(f"**** Saving scores to {score_save_path} ****")
        os.makedirs(os.path.dirname(score_save_path), exist_ok=True)
        with open(score_save_path, "wb") as f:
            pickle.dump({
                "calibration_scores": calibration_scores,
                "test_clean_scores": test_clean_scores,
                "test_poison_scores": test_poison_scores
            }, f)

        with open(os.path.join(os.path.dirname(score_save_path), 'config_used.yaml'), 'w') as f:
            yaml.dump({"backdoor_kwargs": backdoor_kwargs, "score_kwargs": score_kwargs}, f)

    
    if score_name == "mahalanobis":
        # For Mahalanobis, lower scores indicate higher suspicion
        calibration_scores = -calibration_scores
        test_clean_scores = -test_clean_scores
        test_poison_scores = -test_poison_scores

    return calibration_scores, test_clean_scores, test_poison_scores