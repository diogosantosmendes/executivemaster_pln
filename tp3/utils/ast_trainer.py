from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import torch
import torchaudio
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor
from transformers import ASTForAudioClassification, get_cosine_schedule_with_warmup

from .data_loader import load_data
from configs import GENRES, AST_FILES_PER_GENRE, AST_MODEL_NAME, AST_AUDIO_DURATION_SECONDS, AST_EPOCHS, TESTING_SPLIT_RATIOS

class ASTTrainer:
    def __init__(self):
        self.model_path = "models/ast_genre_model"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pass

    def _extract_features(self, audio_paths, labels, audio_start_samples=[0]):
        feature_extractor = ASTFeatureExtractor.from_pretrained(AST_MODEL_NAME)
        features=[]
        
        for idx, path in enumerate(audio_paths):
            waveform, sr = torchaudio.load(path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Resample to 16kHz (AST Requirement)
            if sr != 16000:
                resample = torchaudio.transforms.Resample(sr, 16000)
                waveform = resample(waveform)

            sr = 16000
            target_length = sr * AST_AUDIO_DURATION_SECONDS
            
            if waveform.shape[1] > target_length:
                n_samples = waveform.shape[1]
                for audio_start in audio_start_samples:
                    start_sample = int(n_samples * audio_start)
                    end_sample = int(start_sample + AST_AUDIO_DURATION_SECONDS * sr)
                    end_sample = min(end_sample, waveform.shape[1])
                    sample = waveform[:, start_sample:end_sample]

                    inputs = feature_extractor(
                        sample.squeeze().numpy(),
                        sampling_rate=sr,
                        return_tensors="pt"
                    )

                    features.append({
                        "input_values": inputs.input_values.squeeze(0),
                        "labels": torch.tensor(labels[idx])
                    })
            else:
                padding = target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

                inputs = feature_extractor(
                    waveform.squeeze().numpy(), 
                    sampling_rate=sr, 
                    return_tensors="pt"
                )
                
                features.append({
                    "input_values": inputs.input_values.squeeze(0),
                    "labels": torch.tensor(labels[idx])
                })

        feature_extractor.save_pretrained(self.model_path)
        return features

    def evaluate(self, val_loader):
        self.model.eval()
        preds, refs = [], []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_values=inputs)
                pred = outputs.logits.argmax(dim=-1)

                preds.extend(pred.cpu().numpy())
                refs.extend(labels.cpu().numpy())

        return {
            "acc": accuracy_score(refs, preds),
            "pre": precision_score(refs, preds, average="weighted"),
            "rec": recall_score(refs, preds, average="weighted"),
            "f1": f1_score(refs, preds, average="weighted")
        }

    def train(self, files_per_genre=AST_FILES_PER_GENRE):
        audio_paths, labels = load_data(files_per_genre)
        print(f"Total audio files loaded: {len(audio_paths)}")

        print("--- Extracting Features for Transformer (AST) ---")
        X = self._extract_features(audio_paths, labels)

        print("--- Data Split ---") 
        X_train, X_tmp, _, y_tmp = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        X_val, X_test, _, _ = train_test_split(
            X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
        )

        print("--- Loading Transformer model (AST) ---")
        genre_to_id = {genre: i for i, genre in enumerate(GENRES)}
        id_to_genre = {i: genre for genre, i in genre_to_id.items()}

        self.model = ASTForAudioClassification.from_pretrained(
            AST_MODEL_NAME, 
            num_labels=len(GENRES), 
            ignore_mismatched_sizes=True,
            id2label=id_to_genre,
            label2id=genre_to_id
        )
        self.model.to(self.device)

        print("--- Training Transformer (AST) ---")

        train_loader = DataLoader(X_train, batch_size=2, shuffle=True, pin_memory=True)
        val_loader = DataLoader(X_val, batch_size=2, pin_memory=True)
        
        # Use AdamW optimizer with weight decay and a cosine learning rate scheduler with warmup 
        # to improve training stability and convergence for the AST model.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=1e-4)
        # Use label smoothing in the loss function to help prevent overfitting and improve generalization of the AST model.
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        num_training_steps = AST_EPOCHS * len(train_loader)
        num_warmup_steps = int(0.2 * num_training_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        patience = 4
        counter = 0
        best_val_acc = 0

        scaler = torch.amp.GradScaler()

        for epoch in range(AST_EPOCHS):
            self.model.train()

            for batch in train_loader:
                inputs = batch["input_values"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=self.device.type):
                    outputs = self.model(input_values=inputs)
                    loss = criterion(outputs.logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()

            val_acc = self.evaluate(val_loader)["acc"]
            print(f"Epoch {epoch} done | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.model.save_pretrained(self.model_path)
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping!")
                break

        
        print("--- Evaluating Transformer (AST) ---")
        # reload best model before final evaluation to avoid overfit epoch
        self.load_model()
        test_loader = DataLoader(X_test, batch_size=2, pin_memory=True)
        results = self.evaluate(test_loader)

        print(f"Transformer (AST) Accuracy: {results['acc']:.4f}")
        print(f"Transformer (AST) Precision: {results['pre']:.4f}")
        print(f"Transformer (AST) Recall: {results['rec']:.4f}")
        print(f"Transformer (AST) F1 Score: {results['f1']:.4f}")
        

    def load_model(self):
        self.model = ASTForAudioClassification.from_pretrained(self.model_path)
        self.model.to(self.device)

    def predict(self, file_path):
        features = self._extract_features([file_path], [0], audio_start_samples=TESTING_SPLIT_RATIOS)
        
        self.model.eval()
        with torch.no_grad():
            votes = []
            for feature in features:
                inputs = feature["input_values"].unsqueeze(0).to(self.device)
                logits = self.model(input_values=inputs).logits
                pred = torch.argmax(logits, dim=-1).item()
                votes.append(pred)

            pred = max(set(votes), key=votes.count)
            return GENRES[pred]

