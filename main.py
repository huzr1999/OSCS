# from openbackdoor.defenders import load_defender
# from utils.logger import get_logger
import numpy as np
from tqdm import tqdm
from OSCS.main import OSCS
import numpy as np
from utils.logger import get_logger, init_logger
from utils.metrics import Metrics
from utils.utils import set_config, get_online_test_sequence, get_online_metric_history
from utils.plot import plot_and_save_density_estimations, plot_and_save_scores_histogram, plot_and_save_overall_performance
from scoring.utils import prepare_scores
import yaml
import settings
import logging
import json
import pickle
import os


def run_OSCS_one_trial(calibration_scores, test_scores, test_labels, oscs_kwargs, poison_ratio, log_interval, score_kwargs, **kwargs):
    
    logger = get_logger(__name__)

    # metric = Metrics()

    oscs = OSCS(
        calib_scores=calibration_scores,
        **oscs_kwargs
    )

    decisions = []
    fsr_history = []
    power_history = []
    for step, new_test_score in enumerate(tqdm(test_scores, desc="Processing test samples")):
        decision, current_local_fdr = oscs.update_and_decide(new_test_score)
        decisions.append(decision)

        if step % log_interval == (log_interval - 1):
            logger.debug("="*50)
            logger.debug(f"Test Sample {step}: Score={new_test_score:.4f}, Decision={decision}")
            logger.debug("="*20)
            logger.debug(f"Estimated Alt Proportion={oscs.current_alt_proportion:.4f}, Estimated pd on alt: {oscs.current_pd_on_alt:.4f}, Estimated pd on mix: {oscs.current_pd_on_mix:.4f}, Estimated Local FDR={current_local_fdr:.4f}")
            logger.debug("="*20)
            logger.debug(f"True Alt Proportion={1 - poison_ratio:.4f}")
            logger.debug("="*50)

    fsr_history, power_history = get_online_metric_history(test_labels, decisions)

    plot_and_save_density_estimations(oscs, **oscs_kwargs["rkde_kwargs"], test_X=test_scores, decisions=decisions)
    plot_and_save_scores_histogram(calibration_scores, test_scores[test_labels==1], test_scores[test_labels==0], score_name=score_kwargs['full_name'])

    return fsr_history, power_history




def run_online_trials(num_trials, random_seed, experimental_kwargs, backdoor_kwargs, **kwargs):
    
    fsr_history_all = []
    power_history_all = []
    logger = get_logger(__name__)


    rng = np.random.default_rng(random_seed)
    if experimental_kwargs["method"] in ["OSCS"]:
        calibration_scores, test_clean_scores, test_poison_scores = prepare_scores(backdoor_kwargs, experimental_kwargs['score_kwargs'], rng)
    else:
        raise ValueError(f"Unknown method {experimental_kwargs['method']}")


    for num_trial in range(num_trials):
        logger.info(f"--------------------- Starting trial {num_trial + 1}/{num_trials} ---------------------")

        rng = np.random.default_rng(random_seed + num_trial)
        
        if experimental_kwargs["method"] == "OSCS":
            test_scores, test_Y = get_online_test_sequence(test_clean_scores, test_poison_scores, experimental_kwargs["poison_ratio"], experimental_kwargs["T"], rng)
            fsr_history, power_history = run_OSCS_one_trial(
                calibration_scores, test_scores, test_Y,
                **experimental_kwargs,
                rng=rng
            )
        else:
            raise ValueError(f"Unknown method {experimental_kwargs['method']}")

        logger.info(f"Trial {num_trial + 1} Results: FSR = {fsr_history[-1]:.4f}, Power = {power_history[-1]:.4f}")

        fsr_history_all.append(fsr_history)
        power_history_all.append(power_history)

    avg_fsr_history = np.mean(fsr_history_all, axis=0)
    avg_power_history = np.mean(power_history_all, axis=0)

    std_fsr_history = np.std(fsr_history_all, axis=0)
    std_power_history = np.std(power_history_all, axis=0)

    logger.info("--- All trials completed. ---")
    logger.info(f"Average Final FSR over {num_trials} trials: {avg_fsr_history[-1]:.4f} ± {std_fsr_history[-1]:.4f}")
    logger.info(f"Average Final Power over {num_trials} trials: {avg_power_history[-1]:.4f} ± {std_power_history[-1]:.4f}")

    plot_and_save_overall_performance(avg_fsr_history, std_fsr_history, avg_power_history, std_power_history, experimental_kwargs['oscs_kwargs']['alpha'])

    return {
        "fsr_history_all": fsr_history_all,
        "power_history_all": power_history_all
    }



# %%
if __name__ == "__main__":

    configs = settings.configs
    set_config(configs)

    print(configs)
    RESULTS_PATH = settings.RESULTS_PATH

    init_logger(os.path.join(RESULTS_PATH, "debug.log"), log_level=logging.INFO, log_file_level=logging.INFO)
    logger = get_logger(__name__, logging.DEBUG)

    results = run_online_trials(**configs)


    logger.info(f"Saving results to {RESULTS_PATH}")

    with open(os.path.join(RESULTS_PATH, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    with open(os.path.join(RESULTS_PATH, 'config_used.yaml'), 'w') as f:
        yaml.dump(configs, f)



