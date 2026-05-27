class Metrics:
	def __init__(self):
		self.cumulative_TP = 0
		self.cumulative_FP = 0
		self.cumulative_TN = 0
		self.cumulative_FN = 0
	
	def update(self, true_label, predicted_label):
		if predicted_label == 1:
			if true_label == 1:
				self.cumulative_TP += 1
			else:
				self.cumulative_FP += 1
		else:
			if true_label == 1:
				self.cumulative_FN += 1
			else:
				self.cumulative_TN += 1
	
	def precision(self):
		denom = self.cumulative_TP + self.cumulative_FP
		if denom == 0:
			return 0.0
		return self.cumulative_TP / denom
	
	def recall(self):
		denom = self.cumulative_TP + self.cumulative_FN
		if denom == 0:
			return 0.0
		return self.cumulative_TP / denom
	
	def fsr(self):
		denom = self.cumulative_TP + self.cumulative_FP
		if denom == 0:
			return 0.0
		return self.cumulative_FP / denom