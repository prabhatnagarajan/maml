import torch
from collections import OrderedDict 


class MAMLSupervised:

	def __init__(self, network,
				 meta_train_iterations, num_shots,
				 task_distribution, meta_batch_size, optimizer,
				 subtask_optimize, num_gradient_updates,
				 subtask_loss):
		self.network = network
		self.net_weights = list(self.network.parameters())
		self.meta_train_iterations = meta_train_iterations
		self.K_shot = num_shots
		self.task_distribution = task_distribution
		self.meta_batch_size = meta_batch_size
		self.optimizer = optimizer
		self.subtask_optimize = subtask_optimize
		self.subtask_loss = subtask_loss
		# TODO: num_gradient_updates
		self.num_gradient_updates = num_gradient_updates

	def inner_update(self, task, network_copies, num_gradient_updates):
		network_params = OrderedDict(self.network.named_parameters())
		for _ in range(num_gradient_updates):
	       training_inputs, training_labels = task.sample_dataset(dataset_size=self.K_shot)
	       # TODO: Move to device
	       predictions = self.network.substituted_forward(torch.tensor(training_inputs, dtype=torch.float,requires_grad=True),
	                                                                                                  named_params=network_params)
	       loss = self.subtask_loss(predictions, torch.tensor(training_labels, dtype=torch.float32))
	          gradients = torch.autograd.grad(loss, network_params.values(), create_graph=True)
	       network_params = OrderedDict(
	       (name, param - 0.001 * gradient)
	       for ((name, param), gradient) in zip(network_params.items(), gradients))
		network_copies.append(network_params)

	def inner_update(self, task, network_copies):
		training_inputs, training_labels = task.sample_dataset(dataset_size=self.K_shot)
		# TODO: Move to device
		predictions = network_copy(torch.tensor(training_inputs, dtype=torch.float))
		task_loss_func = self.subtask_optimize(network_copy, predictions,
											   torch.tensor(training_labels, dtype=torch.float32))
		network_copies.append(network_copy)

	def meta_update(self, task_test_batch, network_copies):
		losses = []
		for i in range(len(task_test_batch)):
			inputs, labels = task_test_batch[i]
			predictions = self.network.substituted_forward(torch.tensor(inputs, dtype=torch.float, requires_grad=True),
                                                           named_params=network_copies[i])
			losses.append(self.subtask_loss(predictions, torch.tensor(labels, dtype=torch.float)))
		loss = sum(losses)
		old_params = {}
		for name, params in self.network.named_parameters():
		    old_params[name] = params.clone()
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		self.check_changed(old_params)

	def check_changed(self, old_params):
		# perform update
		for name, params in self.network.named_parameters():
			assert not (old_params[name] == params).all()


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
			print("Completed iteration " + str(iteration))

