'''import recurrent_gpu
import neat
import saving
import numpy as np
import cupy as cp
import time
from numba import njit


def check_inputs_cpu(network, inputs):
    for i in range(inputs.shape[0]):
        network.activate(inputs[i])


@njit
def check_input_njit(network, inputs):
    for i in range(inputs.shape[0]):
        network.activate(inputs[i])


def run_tests(genome, config):
    inputs_np = np.random.uniform(-1, 1, (500, 19))
    inputs_cp = cp.array(inputs_np)

    network_cpu = neat.nn.RecurrentNetwork.create(genome, config)

    start_time = time.time()
    check_inputs_cpu(network_cpu, inputs_np)

    print("---CPU---")
    print(" Time: ", time.time() - start_time)
    print(" Outputs: ", network_cpu.activate(inputs_np[-1]))
    print(" Values: ", network_cpu.values)

    network_gpu = recurrent_gpu.RecurrentNetworkCreator.create(genome, config)

    start_time = time.time()
    check_input_njit(network_gpu, inputs_cp)

    print("---GPU---")
    print(" Time: ", time.time() - start_time)
    print(" Outputs: ", network_gpu.activate(inputs_cp[-1]))
    print(" Values: ", network_gpu.values)


if __name__ == "__main__":
    save_system = saving.SaveSystem(1, "Short-5pw-genome-DXYZ.gz", 8, "Short-5pw-population-DXYZ.gz")
    genome = save_system.load_data("Saves/Genomes/Short-5pw-genome-DXYZ.gz")
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                     neat.DefaultStagnation, "C:\\Users\\vpr16\\PythonProjects\\StockTraderNEATShort\\config_recurrent.txt")

    run_tests(genome, config)
'''