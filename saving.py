import gzip
import pickle
import random
from neat.population import Population
from neat.reporting import BaseReporter
import os
import csv


class SaveSystem(BaseReporter):
    def __init__(self, genome_interval=1, genome_file_path='neat-genome.gz', population_interval=5, population_file_path='neat-population.gz'):
        self.g_interval = genome_interval
        self.g_path = genome_file_path
        self.p_interval = population_interval
        self.p_path = population_file_path
        self.current_generation = None
        self.consecutive_gens = 0

    def start_generation(self, generation):
        self.current_generation = generation
        self.consecutive_gens += 1

    def end_generation(self, config, population, species_set):
        if self.consecutive_gens % self.g_interval == 0:
            best_genome = None
            for key in population:
                if population[key].fitness is not None and (best_genome is None or population[key].fitness > best_genome.fitness):
                    best_genome = population[key]
            self.save_data(best_genome, self.g_path)

        if self.consecutive_gens >= self.p_interval:
            self.save_data((self.current_generation + 1, config, population, species_set, random.getstate()), self.p_path)
            self.consecutive_gens = 0

    @staticmethod
    def make_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    @staticmethod
    def write_to_file(string, path, mode="w"):
        print(f"Writing string to {path}")

        with open(path, mode) as f:
            f.write(string)

    @staticmethod
    def read_from_file(path, mode="r"):
        with open(path, mode) as f:
            return f.read()

    @staticmethod
    def make_csv(header, path, mode="w"):
        with open(path, mode, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    @staticmethod
    def save_to_csv(data, path, mode="w"):
        with open(path, mode, newline="") as f:
            writer = csv.writer(f)
            if mode == "a":
                writer.writerow(data)
            else:
                writer.writerows(data)

    @staticmethod
    def read_from_csv(path, mode="r"):
        with open(path, mode, newline="") as f:
            reader = csv.reader(f)
            return list(reader)

    @staticmethod
    def save_data(data, path, mode="w"):
        print(f"Saving data to {path}")

        with gzip.open(path, mode, compresslevel=5) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_data(path, mode="r"):
        with gzip.open(path, mode) as f:
            return pickle.load(f)

    @staticmethod
    def load_population(path):
        with gzip.open(path) as f:
            generation, config, population, species_set, rndstate = pickle.load(f)
            random.setstate(rndstate)
            return Population(config, (population, species_set, generation))

    @staticmethod
    def delete_file(path):
        if os.path.exists(path):
            os.remove(path)
