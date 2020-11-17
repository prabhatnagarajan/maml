import copy
import torch

def clone_network(network):
	network_copy = copy.deepcopy(network)
	state_dict = network.state_dict()
	network_copy_state_dict = network_copy.state_dict()
	for key in state_dict.keys():
		network_copy_state_dict[key] = torch.clone(state_dict[key])
	# network_copy.to(network.device)
	# TODO: device
	return network_copy
