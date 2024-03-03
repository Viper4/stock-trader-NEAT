import gzip
import pickle
import random
from neat.population import Population
from neat.reporting import BaseReporter


class SaveSystem(BaseReporter):
    def __init__(self, genome_interval=1, genome_file_path='neat-genome.gz', population_interval=5, population_file_path='neat-population.gz'):
        self.g_interval = genome_interval
        self.g_path = genome_file_path
        self.p_interval = population_interval
        self.p_path = population_file_path
        self.current_generation = None

    def start_generation(self, generation):
        self.current_generation = generation

    def end_generation(self, config, population, species_set):
        if self.current_generation % self.g_interval == 0:
            best_genome = None
            for key in population:
                if population[key].fitness is not None and (best_genome is None or population[key].fitness > best_genome.fitness):
                    best_genome = population[key]
            self.save_data(best_genome, self.g_path)

        if self.current_generation % self.p_interval == 0:
            self.save_data((self.current_generation, config, population, species_set, random.getstate()), self.p_path)

    @staticmethod
    def save_data(data, path, mode="w"):
        print(f"Saving data to {path}")

        with gzip.open(path, mode, compresslevel=5) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_data(filepath, mode="r"):
        with gzip.open(filepath, mode) as f:
            return pickle.load(f)

    @staticmethod
    def load_population(filepath):
        with gzip.open(filepath) as f:
            generation, config, population, species_set, rndstate = pickle.load(f)
            random.setstate(rndstate)
            return Population(config, (population, species_set, generation))
