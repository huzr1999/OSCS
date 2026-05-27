from .rkde import RecursiveKDE
import numpy as np
from scipy.stats import gaussian_kde

def calculate_grid_params(calib_scores, grid_range_scale, grid_resolution, **kwargs):
    min_calib_score = np.min(calib_scores)
    max_calib_score = np.max(calib_scores)
    score_range = max_calib_score - min_calib_score
    grid_min = min_calib_score - ((grid_range_scale - 1) // 2) * score_range
    grid_max = max_calib_score + ((grid_range_scale - 1) // 2) * score_range
    num_grids = int((grid_max - grid_min) / grid_resolution) + 1

    bandwidth = 1.06 * np.std(calib_scores) * (len(calib_scores) ** (-1 / 5))
    return grid_min.item(), grid_max.item(), num_grids, bandwidth.item()

class ProportionEstimator:
    def __init__(self, q_calib, lambda_):
        self.lambda_ = lambda_
        self.counts = 0
        self.q_calib = q_calib
        self.p_value_greater_than_lambda_num = 0

    def update(self, new_q):
        
        self.counts += 1

        def G(x):
            return x

        # Here, the new_q is the score of the new test sample and could be further processed to ensure clean samples tend to have higher scores
        new_p_value = (1 + np.sum(G(self.q_calib) <= G(new_q))) / (1 + len(self.q_calib))
        if new_p_value > self.lambda_:
            self.p_value_greater_than_lambda_num += 1

    def estimate(self):
        if self.counts == 0:
            raise ValueError("No samples have been processed yet.")
        return self.p_value_greater_than_lambda_num / (self.counts * (1 - self.lambda_))

class OSCS:
    def __init__(self, calib_scores, alpha, warmup_steps, lambd, bandwidth_kde, rkde_kwargs):
        self.calib_scores = calib_scores

        # calib_scores should be a 1D array-like structure where each element is the score of a calibration sample

        # bandwidth_kde can be a scalar or a string indicating the method for bandwidth selection

        self.kde_estimator = gaussian_kde(calib_scores, bandwidth_kde)

        grid_min, grid_max, num_grids, bandwidth = calculate_grid_params(calib_scores, **rkde_kwargs)

        rkde_kwargs.update({
            "grid_min": grid_min,
            "grid_max": grid_max,
            "num_grids": num_grids,
            "bandwidth": bandwidth
        })


        self.rkde_estimator = RecursiveKDE(**rkde_kwargs)

        self.proportion_estimator = ProportionEstimator(q_calib=self.calib_scores, lambda_=lambd)

        self.alpha = alpha

        self.selection_count = 0 # Record \sum_{i=1}^t delta_i meaning the number of selected samples so far
        self.accumulated_local_fdr = 0 # Record \sum_{i=1}^t delta_i * L_i meaning the accumulated local fdr of selected samples so far

        self.warmup_steps = warmup_steps
        self.current_step = 0

        self.current_alt_proportion = 0
        self.current_pd_on_alt = 0   
        self.current_pd_on_mix = 0

    def update(self, new_test_score):
        self.rkde_estimator.update(new_test_score)
        self.proportion_estimator.update(new_test_score)


    def compute_local_fdr(self, new_test_score):

        # new_test_score should be a scalar

        pd_on_alt = self.kde_estimator(new_test_score)
        # model the P(w|Y=1)
        pd_on_mix = self.rkde_estimator.evaluate(new_test_score)
        # model the P(w)

        alt_proportion = self.proportion_estimator.estimate()
        # model the P(Y=1). This is the "clean sample" is the sample we are interested in selecting, so it is treated as the alternative hypothesis here.

        current_local_fdr_inv = alt_proportion * pd_on_alt / pd_on_mix

        self.current_alt_proportion = alt_proportion
        self.current_pd_on_alt = pd_on_alt.item()
        self.current_pd_on_mix = pd_on_mix.item()

        current_local_fdr_inv = np.clip(current_local_fdr_inv, 0, 1)

        return 1 - current_local_fdr_inv.item()


    def make_decisions(self, current_local_fdr):

        current_fsr = (self.accumulated_local_fdr + current_local_fdr) / (self.selection_count + 1)

        if current_fsr <= self.alpha:
            self.selection_count += 1
            self.accumulated_local_fdr += current_local_fdr
            return 1  # Select the sample
        else:
            return 0  # Do not select the sample
    
    def update_and_decide(self, new_test_score):

        self.update(new_test_score)

        if self.current_step < self.warmup_steps:
            decision = 0  # During warmup, do not select any samples
            current_local_fdr = 0  # No local FDR computed during warmup
        else:
            current_local_fdr = self.compute_local_fdr(new_test_score)
            decision = self.make_decisions(current_local_fdr)

        self.current_step += 1
        return decision, current_local_fdr