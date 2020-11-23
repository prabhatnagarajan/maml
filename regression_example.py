import argparse
from collections import OrderedDict
import os

import matplotlib.colors as mc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
		x = F.linear(inputs, weight=named_params['regressor.lin1.weight'],
					 bias=named_params['regressor.lin1.bias'])
		x = F.relu(x)
		x = F.linear(x, weight=named_params['regressor.lin2.weight'],
					 bias=named_params['regressor.lin2.bias'])
		x = F.relu(x)
		return F.linear(x, weight=named_params['regressor.output.weight'],
						bias=named_params['regressor.output.bias'])

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
	plt.xlim(-6, 6.)
	plt.ylim(-6, 6)
	plt.ylabel("sinusoid(t)")
	plt.xlabel("t")
	plt.legend()
	plt.title(title)
	plt.savefig(filename)

def predict_and_plot(regressor, task, title, file_suffix):
	inputs, labels = task.sample_dataset(40)
	# Before N-updates
	zero_shot_predictions = regressor(torch.tensor(inputs, dtype=torch.float)).detach().numpy()
	plot_true_v_predicted(inputs, labels, zero_shot_predictions,
	                                         plot_type="scatter", filename="0_shot_"+ file_suffix +".png",
	                                         title=title + ": Zero shot")
	networks = []
	maml_trainer.inner_update(task, networks, 10)
	k_shot_predictions = regressor.substituted_forward(torch.tensor(inputs, dtype=torch.float),
	                                                  named_params=networks[0]).detach().numpy()
	plot_true_v_predicted(inputs, labels, k_shot_predictions,
	                      plot_type="scatter",
	                      filename="k_shot_"+ file_suffix +".png",
	                      title=title + ": 10 gradient updates")

def demo(network_path, regressor):
	regressor.load_state_dict(torch.load(network_path))
	# TODO: move things to device
	task_distribution = SinusoidTaskDistribution()
	task = task_distribution.sample_tasks(1)[0]
	predict_and_plot(regressor, task, "Randomly sampled task", "rnd_task")
	task_low = SinusoidTask(x_low=-5.0,
							x_high=5.0,
							amplitude=0.1,
							phase=0.0)
	predict_and_plot(regressor, task_low, "Lowest Extreme task", "low_extreme")
	task_high = SinusoidTask(x_low=-5.0,
							x_high=5.0,
							amplitude=5.0,
							phase=np.pi)
	predict_and_plot(regressor, task_high, "Highest Extreme task", "high_extreme")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--demo", action="store_true", default=False)
	parser.add_argument("--results-dir", type=str, default="results")
	parser.add_argument("--meta-train-iterations", type=int, default=15000)
	parser.add_argument("--meta-step-size", type=float, default=0.001)
	parser.add_argument("--inner-step-size", type=float, default=0.01)
	parser.add_argument("--meta-batch-size", type=int, default=25)
	parser.add_argument("--num-gradient-updates", type=int, default=10)
	parser.add_argument("--num-shots", type=int, default=10)
	parser.add_argument("--device", choices=["gpu", "cuda", "cpu"], default="cpu")
	args = parser.parse_args()
	regressor = MetaLearnedRegressor()
	if args.device == "cuda" or args.device == "gpu":
		assert torch.cuda.is_available()
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")
	# TODO:Move net to device
	meta_optimizer = torch.optim.Adam(regressor.parameters(), lr=args.meta_step_size)
	task_distribution = SinusoidTaskDistribution()
	maml_trainer = maml.SupervisedMAML(network=regressor,
			 					   meta_train_iterations=args.meta_train_iterations,
			 					   inner_step_size=args.inner_step_size,
			 					   num_shots=args.num_shots,
			 					   task_distribution=task_distribution,
			 					   meta_batch_size=args.meta_batch_size,
			 					   optimizer=meta_optimizer,
			 					   subtask_loss=torch.nn.MSELoss(),
			 					   num_gradient_updates=args.num_gradient_updates,
			 					   results_dir=args.results_dir)
	if args.demo:
		demo(os.path.join(args.results_dir, "regressor.pt"), regressor)
	else:
		# TODO, add device
		maml_trainer.train()

