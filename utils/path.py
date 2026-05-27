import os
import settings
from pathlib import Path
from utils.logger import get_logger

def get_newest_directory(base_path: str) -> str | None:
    """
    Finds the directory with the most recent modification time (mtime)
    under the specified base path.

    Args:
        base_path: The root directory to search within.

    Returns:
        The full path of the newest subdirectory, or None if no directories are found.
    """
    
    # 1. Convert the input string to a Path object for easier handling
    base_dir = Path(base_path)

    # 2. Check if the base path exists
    if not base_dir.is_dir():
        # print(f"Error: Base path not found or is not a directory: {base_path}")
        return None

    # 3. Use glob to find all subdirectories one level deep
    #    We check if the path is a directory (is_dir()) and store its path
    subdirs = [p for p in base_dir.iterdir() if p.is_dir()]

    if not subdirs:
        # print(f"No subdirectories found under: {base_path}")
        return None

    # 4. Find the directory with the maximum modification time (mtime)
    #    The 'key' argument in max() tells it how to compare items.
    #    p.stat().st_mtime returns the modification time as a float (seconds since epoch).
    try:
        newest_dir = max(subdirs, key=lambda p: p.stat().st_mtime)
        return str(newest_dir)
    except Exception as e:
        print(f"An error occurred while comparing modification times: {e}")
        return None

def get_inter_dir_string(backdoor_kwargs):

    poison_setting = backdoor_kwargs['attacker']['train']['poison_setting']
    poison_method = backdoor_kwargs['attacker']['train']['poison_method']
    poison_rate = backdoor_kwargs['attacker']['poisoner']['poison_rate']
    poison_dataset_name = backdoor_kwargs['poison_dataset']['name']
    trainer_name = backdoor_kwargs['attacker']['train']['name']
    act_lambd = backdoor_kwargs['attacker']['train']['act_lambd']
    rep_lambd = backdoor_kwargs['attacker']['train']['rep_lambd']
    trainer_name = backdoor_kwargs["attacker"]["train"]["name"].lower()
    num_train_epochs = backdoor_kwargs["attacker"]["train"]["epochs"]

    if trainer_name == "base":
        inter_path = f'{poison_dataset_name}-{poison_setting}-{poison_method}-{poison_rate}-{num_train_epochs}'
    elif trainer_name == "adaptive":
        inter_path = f'{poison_dataset_name}-{poison_setting}-{poison_method}-{poison_rate}-{num_train_epochs}-{trainer_name}-{act_lambd}-{rep_lambd}'
    else:
        raise ValueError(f"Unknown trainer name: {trainer_name}")
    return inter_path

def get_bki_inter_dir_string(backdoor_kwargs):

    poison_setting = backdoor_kwargs['attacker']['train']['poison_setting']
    poison_method = backdoor_kwargs['attacker']['train']['poison_method']
    poison_rate = backdoor_kwargs['attacker']['poisoner']['poison_rate']
    poison_dataset_name = backdoor_kwargs['poison_dataset']['name']
    num_train_epochs = backdoor_kwargs["defender"]["bki_epochs"]
    model_type = backdoor_kwargs["defender"]["bki_model_name"]

    inter_path = f'{model_type}-{poison_dataset_name}-{poison_setting}-{poison_method}-{poison_rate}-{num_train_epochs}'

    return inter_path

def get_cube_inter_dir_string(backdoor_kwargs):

    poison_setting = backdoor_kwargs['attacker']['train']['poison_setting']
    poison_method = backdoor_kwargs['attacker']['train']['poison_method']
    poison_rate = backdoor_kwargs['attacker']['poisoner']['poison_rate']
    poison_dataset_name = backdoor_kwargs['poison_dataset']['name']
    num_train_epochs = backdoor_kwargs["defender"]["cube_epochs"]
    model_type = backdoor_kwargs["defender"]["cube_model_name"]

    inter_path = f'{model_type}-{poison_dataset_name}-{poison_setting}-{poison_method}-{poison_rate}-{num_train_epochs}'

    return inter_path

def get_defender_preds_inter_dir_string(backdoor_kwargs):

    poison_setting = backdoor_kwargs['attacker']['train']['poison_setting']
    poison_method = backdoor_kwargs['attacker']['train']['poison_method']
    poison_rate = backdoor_kwargs['attacker']['poisoner']['poison_rate']
    poison_dataset_name = backdoor_kwargs['poison_dataset']['name']
    num_train_epochs = backdoor_kwargs["attacker"]["train"]["epochs"]

    defender_name = backdoor_kwargs["defender"]["name"].lower()

    inter_path = f'{poison_dataset_name}-{poison_setting}-{poison_method}-{poison_rate}-{num_train_epochs}-{defender_name}'

    return inter_path
    
def get_victim_basepath(backdoor_kwargs):
    model_base_base_path = backdoor_kwargs['attacker']['train']['save_base_path']

    victim_type = backdoor_kwargs['victim']['model_name']
    inter_path = get_inter_dir_string(backdoor_kwargs)

    return os.path.join(model_base_base_path, victim_type, inter_path)

def get_scores_basepath(backdoor_kwargs, score_kwargs):

    score_to_save_path = {
        'md': "mahalanobis_scores",
        'badacts': "badacts_scores",
    }

    scores_base_base_path = score_kwargs['scores_save_base_path']

    victim_type = backdoor_kwargs['victim']['model_name']
    score_save_dir_name = score_to_save_path[score_kwargs['name']]
    inter_path = get_inter_dir_string(backdoor_kwargs)

    return os.path.join(scores_base_base_path, victim_type, score_save_dir_name, inter_path)

def get_bki_basepath(backdoor_kwargs):
    base_base_path = backdoor_kwargs["defender"]["bki_model_saved_base_path"]
    victim_type = backdoor_kwargs['victim']['model_name']
    inter_path = get_bki_inter_dir_string(backdoor_kwargs)
    base_path = os.path.join(base_base_path, victim_type, inter_path)
    return base_path

def get_cube_basepath(backdoor_kwargs):
    base_base_path = backdoor_kwargs["defender"]["cube_model_saved_base_path"]
    victim_type = backdoor_kwargs['victim']['model_name']
    inter_path = get_cube_inter_dir_string(backdoor_kwargs)
    base_path = os.path.join(base_base_path, victim_type, inter_path)
    return base_path

def get_baseline_basepath(backdoor_kwargs):

    base_path = backdoor_kwargs['save_defender_results_base_path']
    victim_type = backdoor_kwargs['victim']['model_name']
    inter_path = get_defender_preds_inter_dir_string(backdoor_kwargs)

    base_path =  os.path.join(base_path, victim_type, inter_path)
    return base_path

def get_victim_load_path(backdoor_kwargs):

    logger = get_logger(__name__)
    base_path = get_victim_basepath(backdoor_kwargs)
    lastest_dir_path = get_newest_directory(base_path)

    # if lastest_dir_path is None:
    #     logger.info(f"No saved model found under: {base_path}")
    
    return lastest_dir_path

def get_victim_save_path(backdoor_kwargs):

    base_path = get_victim_basepath(backdoor_kwargs)
    save_path = os.path.join(base_path, f'{settings.timestamp_str}')
    return save_path


def get_scores_load_path(backdoor_kwargs, score_kwargs):

    logger = get_logger(__name__)
    base_path = get_scores_basepath(backdoor_kwargs, score_kwargs)
    lastest_dir_path = get_newest_directory(base_path)

    if lastest_dir_path is None:
        # logger.info(f"No saved scores found under: {base_path}")
        return None

    score_name = score_kwargs['full_name']
    lastest_score_path = os.path.join(lastest_dir_path, f"{score_name}_scores.pkl")

    return lastest_score_path

def get_scores_save_path(backdoor_kwargs, score_kwargs):

    base_path = get_scores_basepath(backdoor_kwargs, score_kwargs)

    score_name = score_kwargs['full_name']
    save_path = os.path.join(base_path, f'{settings.timestamp_str}', f"{score_name}_scores.pkl")

    return save_path

def get_bki_victim_load_path(backdoor_kwargs):

    base_path = get_bki_basepath(backdoor_kwargs)
    lastest_dir_path = get_newest_directory(base_path)
    
    return lastest_dir_path

def get_bki_victim_save_path(backdoor_kwargs):

    base_path = get_bki_basepath(backdoor_kwargs)
    save_path = os.path.join(base_path, f'{settings.timestamp_str}')
    return save_path

def get_cube_victim_load_path(backdoor_kwargs):

    base_path = get_cube_basepath(backdoor_kwargs)
    lastest_dir_path = get_newest_directory(base_path)
    
    return lastest_dir_path

def get_cube_victim_save_path(backdoor_kwargs):

    base_path = get_cube_basepath(backdoor_kwargs)
    save_path = os.path.join(base_path, f'{settings.timestamp_str}')
    return save_path

def get_baseline_preds_load_path(backdoor_kwargs):
    
    base_path = get_baseline_basepath(backdoor_kwargs)
    lastest_dir_path = get_newest_directory(base_path)

    return lastest_dir_path

def get_baseline_preds_save_path(backdoor_kwargs):

    base_path = get_baseline_basepath(backdoor_kwargs)
    save_path = os.path.join(base_path, f'{settings.timestamp_str}')
    return save_path

