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
            self.save_genome(best_genome)

        if self.current_generation % self.p_interval == 0:
            self.save_population(config, population, species_set, self.current_generation)

    def save_genome(self, genome):
        print("Saving genome to {0}".format(self.g_path))

        with gzip.open(self.g_path, 'w', compresslevel=5) as f:
            pickle.dump(genome, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_population(self, config, population, species_set, generation):
        print("Saving population to {0}".format(self.p_path))

        with gzip.open(self.p_path, 'w', compresslevel=5) as f:
            data = (generation, config, population, species_set, random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def restore_genome(file_path):
        with gzip.open(file_path) as f:
            return pickle.load(f)

    @staticmethod
    def restore_population(file_path):
        with gzip.open(file_path) as f:
            generation, config, population, species_set, rndstate = pickle.load(f)
            random.setstate(rndstate)
            return Population(config, (population, species_set, generation))
