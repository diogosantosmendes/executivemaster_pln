import glob
import joblib
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .data_loader import load_data
from configs import GENRES, TRADITIONAL_ML_FILES_PER_GENRE, TRADITIONAL_ML_AUDIO_DURATION_SECONDS, TESTING_SPLIT_RATIOS

class TraditionalMLTrainer:
    def __init__(self, model):
        self.model = model
        self.model_name = f"models/{model.__class__.__name__}_genre_model.pkl"
        pass

    def _extract_mfcc(self, file_list, audio_start_samples=[0]):
        features = []
        for f in file_list:
            y, sr = librosa.load(f, sr=None)

            n_samples = len(y)
            for audio_start in audio_start_samples:
                start_sample = int(n_samples * audio_start)
                end_sample = int(start_sample + TRADITIONAL_ML_AUDIO_DURATION_SECONDS * sr)
                end_sample = min(end_sample, len(y))

                sample = y[start_sample:end_sample]

                mfcc = librosa.feature.mfcc(y=sample, sr=sr, n_mfcc=20)
                features.append(np.mean(mfcc.T, axis=0))

        return np.array(features)


    def train(self, files_per_genre=TRADITIONAL_ML_FILES_PER_GENRE):
        audio_paths, labels = load_data(files_per_genre)
        print(f"Total audio files loaded: {len(audio_paths)}")

        print("--- Data Split ---")
        train_paths, test_paths, y_train, y_test = train_test_split(
            audio_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )

        print("--- Extracting Features for Traditional ML ---")
        X_train = self._extract_mfcc(train_paths)
        X_test = self._extract_mfcc(test_paths)

        print("--- Training with Traditional ML (Random Forest) ---")
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, self.model_name)

        print("--- Evaluating Traditional ML ---")
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        print(f"Traditional ML Accuracy: {acc:.4f}")
        print(f"Traditional ML Precision: {pre:.4f}")
        print(f"Traditional ML Recall: {rec:.4f}")
        print(f"Traditional ML F1 Score: {f1:.4f}")


    def load_model(self):
        self.model = joblib.load(self.model_name)


    def predict(self, file_path):
        features = self._extract_mfcc([file_path], audio_start_samples=TESTING_SPLIT_RATIOS)
        gender_idx = self.model.predict(features)

        values, counts = np.unique(np.array(gender_idx), return_counts=True)
        most_voted = values[np.argmax(counts)]
        return GENRES[int(most_voted)]

