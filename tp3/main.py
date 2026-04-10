import glob
import os
from sklearn.ensemble import RandomForestClassifier
from utils import TraditionalMLTrainer, ASTTrainer, play_music

def main():
    
    traditional_ml_trainer = TraditionalMLTrainer(RandomForestClassifier(n_estimators=100, random_state=42))
    # traditional_ml_trainer.train()
    traditional_ml_trainer.load_model()

    ast_trainer = ASTTrainer()
    # ast_trainer.train()
    ast_trainer.load_model()


    folder_path = os.path.join("musics", "*.mp3")
    files = glob.glob(folder_path)

    with open("./tests.csv", mode="w") as results_file:
        for f in files:
            song_name = f.split("\\")[-1][:-4]
            print(f"Predicting for: {song_name}")
            
            ml_prediction = traditional_ml_trainer.predict(f)
            ast_prediction = ast_trainer.predict(f)
            
            results_file.write(f"\"{song_name}\", \"{ml_prediction}\", \"{ast_prediction}\"\n")
            
    results_file.close()
    

if __name__ == "__main__":
    main()