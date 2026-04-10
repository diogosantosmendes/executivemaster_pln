import glob
import os
from configs import DATASET_FOLDER, GENRES

def load_data(files_per_genre):
    audio_paths = []
    labels = []
    
    genre_to_id = {genre: i for i, genre in enumerate(GENRES)}

    for genre in GENRES:
        folder_path = os.path.join(DATASET_FOLDER, genre, "*.wav")
        files = glob.glob(folder_path)

        entries = list(range(len(files)))
        jump = len(entries) // files_per_genre if files_per_genre > 0 else 1
        count = 0
        
        for idx in entries[::jump]:
            file = files[idx]
            if os.path.getsize(file) > 0:  # Integrity check: ensure file is not empty
                audio_paths.append(file)
                labels.append(genre_to_id[genre])
                count += 1
                if count >= files_per_genre:
                    break
            

    return audio_paths, labels