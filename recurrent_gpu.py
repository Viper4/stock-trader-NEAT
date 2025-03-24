'''from neat.graphs import required_for_output
from neat.six_util import itervalues, iteritems
from numba import int32, float64, cuda
from numba.experimental import jitclass
import cupy as cp


spec = [
    ("num_inputs", int32),
    ("num_outputs", int32),
    ("biases", float64[:]),
    ("responses", float64[:]),
    ("weights", float64[:, :]),
    ("num_nodes", int32),
    ("values", float64[:, :]),
    ("active", int32),
]


@jitclass(spec=spec)
class RecurrentNetworkGPU(object):
    def __init__(self, num_inputs, num_outputs, biases, responses, weights):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.biases = biases
        self.responses = responses
        self.weights = weights

        self.num_nodes = self.biases.shape[0]
        self.values = cp.zeros((2, self.num_nodes), dtype=cp.float64)  # Double buffer for values
        self.active = 0

    def reset(self):
        self.values.fill(0.0)
        self.active = 0

    def activate(self, inputs):
        if inputs.shape[0] != self.num_inputs:
            raise RuntimeError(f"Expected {self.num_inputs} inputs, got {inputs.shape[0]}")

        for i in range(inputs.shape[0]):
            self.values[self.active][i] = inputs[i]
            self.values[1 - self.active][i] = inputs[i]

        for i in range(self.num_inputs, self.num_nodes):
            s = 0.0
            # For each input connection to this node
            for j in range(self.weights[i].size):
                s += self.values[self.active][i - self.weights[i].size + j] * self.weights[i, j]
            self.values[1 - self.active][i] = cp.tanh(self.biases[i] + self.responses[i] * s)

        outputs = self.values[1 - self.active][self.num_nodes - self.num_outputs:]
        self.active = 1 - self.active

        return outputs


class RecurrentNetworkCreator(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals

        self.values = [{}, {}]
        for v in self.values:
            for k in inputs + outputs:
                v[k] = 0.0

            for node, ignored_activation, ignored_aggregation, ignored_bias, ignored_response, links in self.node_evals:
                v[node] = 0.0
                for i, w in links:
                    v[i] = 0.0
        self.active = 0

    def reset(self):
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        ivalues = self.values[self.active]
        ovalues = self.values[1 - self.active]
        self.active = 1 - self.active

        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v
            ovalues[i] = v

        for node, activation, aggregation, bias, response, links in self.node_evals:
            node_inputs = [ivalues[i] * w for i, w in links]
            s = aggregation(node_inputs)
            ovalues[node] = activation(bias + response * s)

        return [ovalues[i] for i in self.output_nodes]

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

        remapped_nodes = {}
        biases = cp.zeros(len(node_inputs), dtype=cp.float64)
        responses = cp.zeros(biases.shape[0], dtype=cp.float64)
        max_connections = 0
        iter_node_inputs = iteritems(node_inputs)
        i = 0
        for node_key, inputs in iter_node_inputs:
            remapped_nodes[node_key] = i
            node = genome.nodes[node_key]
            biases[i] = node.bias
            responses[i] = node.response
            max_connections = max(max_connections, len(inputs))
            i += 1

        weights = cp.zeros((biases.shape[0], max_connections), dtype=cp.float64)

        for node_key, inputs in iter_node_inputs:
            for i, w in inputs:
                weights[remapped_nodes[node_key], i] = w

        return RecurrentNetworkGPU(len(genome_config.input_keys), len(genome_config.output_keys), biases, responses, weights)
'''