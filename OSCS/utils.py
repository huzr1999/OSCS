import numpy as np

def single_gaussian_pdf(mu, sigma):

	def pdf(x):
		return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
	
	return pdf

def mix_gaussian_pdf(mu_1, sigma_1, mu_2, sigma_2, alt_proportion_test, **kwargs):

	def pdf(x):
		gaussian_1 = single_gaussian_pdf(mu_1, sigma_1)
		gaussian_2 = single_gaussian_pdf(mu_2, sigma_2)
		return alt_proportion_test * gaussian_1(x) + (1 - alt_proportion_test) * gaussian_2(x)
	
	return pdf