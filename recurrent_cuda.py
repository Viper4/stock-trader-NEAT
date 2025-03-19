from neat.graphs import required_for_output
from neat.six_util import itervalues, iteritems
import torch


class RecurrentNetwork(object):
    def __init__(self, input_nodes, output_nodes, node_evals):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.node_evals = node_evals
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_nodes = max(max(input_nodes, default=0), max(output_nodes, default=0), *(node for node, _, _, _, _, _ in node_evals)) + 1

        self.values = torch.zeros((2, num_nodes), device=self.device)
        self.active = 0

    def activate(self, inputs):
        ivalues = self.values[self.active]
        ovalues = self.values[1 - self.active]
        self.active = 1 - self.active

        # Set input values
        input_tensor = torch.tensor(inputs, device=self.device, dtype=torch.float32)
        ivalues[self.input_nodes] = input_tensor
        ovalues[self.input_nodes] = input_tensor

        # Process nodes
        for node, _, _, bias, response, links in self.node_evals:
            indices, weights = zip(*links) if links else ([], [])
            indices = torch.tensor(indices, device=self.device, dtype=torch.long)
            weights = torch.tensor(weights, device=self.device, dtype=torch.float32)

            node_inputs = (ivalues[indices] * weights).sum()
            ovalues[node] = torch.tanh(bias + response * node_inputs)

        return ovalues[self.output_nodes].tolist()

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a RecurrentNetwork). """
        genome_config = config.genome_config
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

        # Gather inputs and expressed connections.
        node_inputs = {}
        for cg in itervalues(genome.connections):
            if not cg.enabled:
                continue

            i, o = cg.key
            if o not in required and i not in required:
                continue

            if o not in node_inputs:
                node_inputs[o] = [(i, cg.weight)]
            else:
                node_inputs[o].append((i, cg.weight))

        node_evals = []
        for node_key, inputs in iteritems(node_inputs):
            node = genome.nodes[node_key]
            activation_function = genome_config.activation_defs.get(node.activation)
            aggregation_function = genome_config.aggregation_function_defs.get(node.aggregation)
            node_evals.append((node_key, activation_function, aggregation_function, node.bias, node.response, inputs))

        return RecurrentNetwork(genome_config.input_keys, genome_config.output_keys, node_evals)
