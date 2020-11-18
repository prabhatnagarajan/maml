from pdb import set_trace
import utils
import torch

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

	def inner_update(self, task, network_copies):
		training_inputs, training_labels = task.sample_dataset(dataset_size=self.K_shot)
		network_copy = utils.clone_network(self.network)
		# TODO: Move to device
		predictions = network_copy(torch.tensor(training_inputs, dtype=torch.float))
		task_loss_func = self.subtask_optimize(network_copy, predictions,
											   torch.tensor(training_labels, dtype=torch.float32))
		network_copies.append(network_copy)

	def meta_update(self, task_test_batch, network_copies):
		losses = []
		for i in range(len(task_test_batch)):
			inputs, labels = task_test_batch[i]
			network_copy = network_copies[i]
			predictions = network_copy(torch.tensor(inputs, dtype=torch.float))
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
		    if (old_params[name] == params).all():
		        print("True")
	def train(self):
		for iteration in range(self.meta_train_iterations):
			# sample training tasks
			task_batch = self.task_distribution.sample_tasks(num_tasks=self.meta_batch_size)
			assert len(task_batch) == self.meta_batch_size
			# for all tasks in tasks
			network_copies = []
			for task in task_batch:
				self.inner_update(task, network_copies)
			task_test_batch = []
			for task in task_batch:
				testing_inputs, testing_labels = task.sample_dataset(dataset_size=self.K_shot)
				task_test_batch.append((testing_inputs, testing_labels))
			self.meta_update(task_test_batch, network_copies)
			print("Completed iteration " + str(iteration))

