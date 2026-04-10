This project aims to compare classifiers by genre for songs.
Its using the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

### To run
(on Windows)
```
py -3.11 -m venv torch-env
.\torch-env\Scripts\Activate.ps1
pip install -r requirements.txt
```


## Results

#### Using Random Forest as Traditional ML example:
with 500 audio files  (50/genre):
```
    Accuracy:   0.6800
    Precision:  0.6902
    Recall:     0.6800
    F1 Score:   0.6680
```
with 700 audio files  (70/genre):
```
    Accuracy:   0.6571
    Precision:  0.6518
    Recall:     0.6571
    F1 Score:   0.6439
```
with 990 audio files  (99/genre):
```
    Accuracy:   0.5707
    Precision:  0.5636
    Recall:     0.5707
    F1 Score:   0.5622
```

#### Using AST (Audio Spectrogram Transformer):

with 100 audio files (10/genre):
```
    Accuracy:   0.9000
    Precision:  0.8500
    Recall:     0.9000
    F1 Score:   0.8667
```
with 150 audio files (15/genre):
```
    Accuracy:   0.9333
    Precision:  0.9667
    Recall:     0.9333
    F1 Score:   0.9333
```
with 200 audio files (20/genre):
```
    Accuracy:   0.8095
    Precision:  0.8889
    Recall:     0.8095
    F1 Score:   0.8159
```


Future work

check metrics by genre 