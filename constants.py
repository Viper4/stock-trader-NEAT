import os


PROJECT_DIR = os.path.dirname(__file__)
SAVE_DIR = PROJECT_DIR + "\\Saves\\"
GENOME_DIR = SAVE_DIR + "Genomes\\"
LOG_DIR = SAVE_DIR + "Logs\\"
POPULATION_DIR = SAVE_DIR + "Populations\\"
TRAINING_DIR = SAVE_DIR + "TrainingData\\"
VALIDATION_DIR = SAVE_DIR + "ValidationData\\"
VALUES_DIR = SAVE_DIR + "Values\\"
CONFIG_PATH = os.path.join(PROJECT_DIR, "config_recurrent.txt")
SETTINGS_PATH = os.path.join(PROJECT_DIR, "settings.json")
