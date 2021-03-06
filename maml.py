import os
from collections import OrderedDict 

import torch


class SupervisedMAML:

	def __init__(self, network,
				 meta_train_iterations,
				 num_shots,
				 inner_step_size,
				 task_distribution,
				 meta_batch_size,
				 optimizer,
				 num_gradient_updates,
				 subtask_loss,
				 results_dir):
		self.network = network
		self.meta_train_iterations = meta_train_iterations
		self.inner_step_size = inner_step_size
		self.K_shot = num_shots
		self.task_distribution = task_distribution
		self.meta_batch_size = meta_batch_size
		self.optimizer = optimizer
		self.subtask_loss = subtask_loss
		self.num_gradient_updates = num_gradient_updates
		self.results_dir = results_dir
		self.meta_losses = []

	def inner_update(self, task, network_copies, num_gradient_updates):
		network_params = OrderedDict(self.network.named_parameters())
		for _ in range(num_gradient_updates):
			training_inputs, training_labels = task.sample_dataset(dataset_size=self.K_shot)
			# TODO: Move to device
			predictions = self.network.substituted_forward(torch.tensor(training_inputs, dtype=torch.float),
			                                                                                          named_params=network_params)
			loss = self.subtask_loss(predictions, torch.tensor(training_labels, dtype=torch.float32))
			gradients = torch.autograd.grad(loss, network_params.values(), create_graph=True)
			network_params = OrderedDict((name, param - self.inner_step_size * gradient)
				for ((name, param), gradient) in zip(network_params.items(), gradients))
		network_copies.append(network_params)

	def meta_update(self, task_test_batch, network_copies):
		losses = []
		for i in range(len(task_test_batch)):
			inputs, labels = task_test_batch[i]
			predictions = self.network.substituted_forward(torch.tensor(inputs, dtype=torch.float),
                                                           named_params=network_copies[i])
			inner_task_loss = self.subtask_loss(predictions, torch.tensor(labels, dtype=torch.float))
			inner_task_loss.backward(retain_graph=True)
			losses.append(inner_task_loss)
		self.optimizer.zero_grad()
		meta_loss = sum(losses)
		old_params = {}
		for name, params in self.network.named_parameters():
		    old_params[name] = params.clone()
		meta_loss.backward()
		self.optimizer.step()
		self.meta_losses.append(meta_loss.detach().numpy())
		self.check_changed(old_params)

	def check_changed(self, old_params):
		# perform update
		for name, params in self.network.named_parameters():
			if (old_params[name] == params).all():
				print("Issue!")
			# assert not (old_params[name] == params).all()

	def housekeep(self):
		# save network
		self.save_network()
		#: Plot meta-losses

	def save_network(self):
		os.makedirs(self.results_dir, exist_ok=True)
		torch.save(self.network.state_dict(),
				   os.path.join(self.results_dir,"network.pt"))

	def train(self):
		for iteration in range(self.meta_train_iterations):
			# sample training tasks
			task_batch = self.task_distribution.sample_tasks(num_tasks=self.meta_batch_size)
			assert len(task_batch) == self.meta_batch_size
			# for all tasks in tasks
			network_copies = []
			for task in task_batch:
				self.inner_update(task, network_copies, self.num_gradient_updates)
			task_test_batch = []
			for task in task_batch:
				testing_inputs, testing_labels = task.sample_dataset(dataset_size=self.K_shot)
				task_test_batch.append((testing_inputs, testing_labels))
			self.meta_update(task_test_batch, network_copies)
			if self.meta_train_iterations <= 100 or (iteration + 1) % int(self.meta_train_iterations / 100) == 0:
				print("Completed meta iteration " + str(iteration + 1))
		self.housekeep()

