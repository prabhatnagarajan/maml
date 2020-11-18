import argparse
from collections import OrderedDict

import matplotlib.colors as mc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace

import maml

class MetaLearnedRegressor(nn.Module):

	def __init__(self):
       super(MetaLearnedRegressor, self).__init__()
       self.regressor = torch.nn.Sequential(OrderedDict([
       ('lin1', nn.Linear(1, 40)),
       ('relu1', nn.ReLU()),
       ('lin2', nn.Linear(40, 40)),
       ('relu2', nn.ReLU()),
       ('output', nn.Linear(40, 1))]))

	def forward(self, inputs):
	    return self.regressor(inputs)

	def substituted_forward(self, inputs, named_params):
       x = F.linear(inputs, weight=named_params['regressor.lin1.weight'], bias=named_params['regressor.lin1.bias'])
       x = F.relu(x)
       x = F.linear(x, weight=named_params['regressor.lin2.weight'], bias=named_params['regressor.lin2.bias'])
       x = F.relu(x)
       return F.linear(x, weight=named_params['regressor.output.weight'], bias=named_params['regressor.output.bias'])

class SinusoidTask:
	def __init__(self, x_low, x_high,
				 amplitude, phase):
		self.x_low = x_low
		self.x_high = x_high
		self.amplitude = amplitude
		self.phase = phase


	def sample_dataset(self, dataset_size):
		inputs = np.random.uniform(self.x_low, self.x_high, size=(dataset_size, 1))
		labels = self.sinusoid(inputs)
		return inputs, labels

	def sinusoid(self, t):
		return self.amplitude * np.sin(t - self.phase)

class SinusoidTaskDistribution:

	def __init__(self, x_low= -5.0, x_high=5.0,
				 amplitude_low=0.1, amplitude_high=5.0,
				 phase_low=0.0, phase_high=np.pi):
		self.x_low = x_low
		self.x_high = x_high
		x_values = np.random.uniform(x_low, x_high, size=30)

		self.amplitude_low = amplitude_low
		self.amplitude_high = amplitude_high

		self.phase_low = amplitude_high
		self.phase_high = phase_high

	def sample_tasks(self, num_tasks):
		amplitude = np.random.uniform(self.amplitude_low, self.amplitude_high)
		phase = np.random.uniform(self.phase_low, self.phase_high)
		task_batch = []
		for i in range(num_tasks):
			amplitude = np.random.uniform(self.amplitude_low, self.amplitude_high)
			phase = np.random.uniform(self.phase_low, self.phase_high)
			task = SinusoidTask(x_low= self.x_low, x_high=self.x_high,
					 amplitude=amplitude, phase=phase)
			task_batch.append(task)
		return task_batch

	def loss(self, predictions, labels):
		return nn.MSELoss(predictions, labels)


def parse_plot_func(plot_func):
	assert plot_func in ("plot", "scatter")
	if plot_func == "scatter":
		return plt.scatter
	if plot_func == "plot":
		return plt.plot

def plot_true_v_predicted(inputs, labels, predictions, plot_type="plot", label=None,
                          filename="plot.png",
                          title="True vs. Predicted"):
	plt.clf()
	axes = plt.gca()
	plot_func = parse_plot_func(plot_type)
	plot_func(inputs,labels,label="True")
  plot_func(inputs,predictions,label="Predictions")
  plt.ylabel("sinusoid(t)")
  plt.xlabel("t")
  plt.legend()
  plt.title(title)
	plt.savefig(filename)


def subtask_optimize(model, predictions, outputs):
	inner_lr = 0.001
	inner_optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr)
	loss_func = torch.nn.MSELoss()
	loss = loss_func(predictions, outputs)
	inner_optimizer.zero_grad()
	loss.backward()
	inner_optimizer.step()
	return loss_func

def demo(network_path):
	regressor.load_state_dict(torch.load(network_path))
	# TODO: move things to device
	task_distribution = SinusoidTaskDistribution()
	task = task_distribution.sample_tasks(1)[0]
	inputs, labels = task.sample_dataset(40)
	# Before N-updates
	zero_shot_predictions = regressor(torch.tensor(inputs, dtype=torch.float)).detach().numpy()
	plot_true_v_predicted(inputs, labels, zero_shot_predictions,
	                                         plot_type="scatter", filename="0_regressed.png",
	                                         title="Zero shot Predictions")

	networks = []
	maml_trainer.inner_update(task, networks, 1)
	k_shot_predictions = regressor.substituted_forward(torch.tensor(inputs, dtype=torch.float),
	                                                  named_params=networks[0]).detach().numpy()
	plot_true_v_predicted(inputs, labels, zero_shot_predictions,
	                                         plot_type="scatter", filename="1_regressed.png",
	                                         title="K shot Predictions")

if __name__ == '__main__':
	print("Regressing...")
	# plot(x=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
	# 	 y=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
	# 	 plot_type="scatter")
	task_distribution = SinusoidTaskDistribution()

	# y_values = [sinusoid(x, amplitude, phase) for x in x_values]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	regressor = MetaLearnedRegressor()
	# Move net to device


	meta_step_size = 0.001
	meta_optimizer = torch.optim.Adam(regressor.parameters(), lr=meta_step_size)
	parser = argparse.ArgumentParser()
	parser.add_argument("--demo", action="store_true", default=False)
	args = parser.parse_args()
	maml_trainer = maml.MAMLSupervised(network=regressor,
			 					   meta_train_iterations=15000,
			 					   num_shots=10,
			 					   task_distribution=task_distribution,
			 					   meta_batch_size=25,
			 					   optimizer=meta_optimizer,
			 					   subtask_optimize=subtask_optimize,
			 					   subtask_loss=torch.nn.MSELoss(),
			 					   num_gradient_updates=10)
	if args.demo:
		regressor.load_state_dict(torch.load("regressor.pt"))
		# TODO: move things to device
		task = task_distribution.sample_tasks(1)[0]
		inputs, labels = task.sample_dataset(40)
		networks = []
		maml_trainer.inner_update(task, networks)
		predictions = networks[0](torch.tensor(inputs, dtype=torch.float)).detach().numpy()

		plot_true_v_predicted(inputs, labels, predictions, plot_type="scatter", filename="regressed.png")
	else:
		# TODO, add device
		maml_trainer.train()
		torch.save(regressor.state_dict())

