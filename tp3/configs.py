# ---------------------------------
# Global configs
# ---------------------------------
# dataset folder path
DATASET_FOLDER = "genres_original"
# list of genres (must match the folder names in the dataset)
GENRES = ("blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock")
# testing split ratios to evaluate model performance
TESTING_SPLIT_RATIOS = [0.05, 0.5, 0.7, 0.8, 0.9]

# ---------------------------------
# Traditional ML configs
# ---------------------------------

# filename for the saved model

# number of files to load per genre for Traditional ML (to avoid imbalance)
TRADITIONAL_ML_FILES_PER_GENRE = 99
# audio duration in seconds for Traditional ML input
TRADITIONAL_ML_AUDIO_DURATION_SECONDS = 30

# ---------------------------------
# AST configs
# ---------------------------------

# pretrained AST model name (from Hugging Face)
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
# number of files to load per genre for AST training
AST_FILES_PER_GENRE = 20
# audio duration in seconds for AST input (AST requires fixed-length inputs, is recommended 10 seconds)
AST_AUDIO_DURATION_SECONDS = 10
# number of epochs to train the AST model
AST_EPOCHS = 10
