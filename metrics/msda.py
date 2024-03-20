import torch
import numpy as np

def euclidean(x1,x2):
	return ((x1-x2)**2).sum().sqrt()

def k_moment(*outputs, k=1):
	outputs = [(output**k).mean(0) for output in outputs]
	return  sum([euclidean(outputs[row], outputs[col]) for row,col in zip(*np.triu_indices(len(outputs),1))])

def moment(*outputs, params=5):
	outputs = [output-output.mean(0) for output in outputs]
	reg_info = k_moment(*outputs)
	for i in range(params-1):
		reg_info += k_moment(*outputs, k=i+2)	
	return reg_info/(params+1)

