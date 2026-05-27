import torch
from utils.utils import split_dataset_in_half
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

@torch.no_grad()
def get_attribution(victim, sample):

	activations = []
	input_tensor = victim.tokenizer.encode(sample, add_special_tokens=True)
	input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(victim.device)
	outputs = getattr(victim.plm, victim.plm.base_model_prefix).forward(input_tensor, output_hidden_states=True)

	for i, f in enumerate(outputs.hidden_states):
		if i > 0:
			activations.extend(f[:, 0, :].view(-1).detach().cpu().numpy().tolist())

	return activations

def feature_process(benign_texts, victim):

	clean_dev_attribution = []
	for t, l, _ in tqdm(benign_texts, desc="Calculating activation for clean dev set"):
		attribution = get_attribution(victim, t)
		clean_dev_attribution.append(attribution)

	return np.array(clean_dev_attribution)

def calcualte_scores(dataset, victim, norm_para, delta):

	scores = []
	for t, l, _ in tqdm(dataset, desc="Calculating BadActs scores"):
		attribution = get_attribution(victim, t)
		pdf = []
		for i, a in enumerate(attribution):
			mu, sigma = norm_para[i]
			pdf.append(int((mu - sigma * delta) <= a <= (mu + sigma * delta)))
		scores.append(np.mean(pdf))
	return np.array(scores)

def badacts_for_datasets(dev_dataset, test_clean_dataset, test_poison_dataset, victim, rng, delta):

	victim.plm.eval()

	dev_dataset_1, dev_dataset_2 = split_dataset_in_half(dev_dataset, rng)
	clean_dev_attribution = feature_process(dev_dataset_1, victim)

	norm_para = []
	for i in range(clean_dev_attribution.shape[1]):
		column_data = clean_dev_attribution[:, i]
		mu, sigma = norm.fit(column_data)
		norm_para.append((mu,sigma))

	calibration_scores = calcualte_scores(dev_dataset_2, victim, norm_para, delta)
	test_clean_scores = calcualte_scores(test_clean_dataset, victim, norm_para, delta)
	test_poison_scores = calcualte_scores(test_poison_dataset, victim, norm_para, delta)

	return calibration_scores, test_clean_scores, test_poison_scores