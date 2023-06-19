# EEG_classification

For corrent work Python 3.7 is required. 

## Datasets used:

1) Innopolis Visual Necker Cube dataset: File "EEG_data.mat" from folder https://disk.yandex.ru/d/ds1JD6GZglkDrg. Use ```to_csv.m``` script for for generating .csv files for each trial. .csv files are needed for further processing.

2) Innopolis Motor dataset: File "sub_all.mat" from folder https://disk.yandex.ru/d/ds1JD6GZglkDrg. Scripts for generating .csv files from .mat file are present in ```Jupyter_Notebooks/Main_notebook.ipynb```.

3) BCI_competition_IV dataset: https://cloud.mail.ru/public/UTFX/16Zpmn92b.

## The structure of this repository:

* ```Jupyter_Notebooks``` - folder contating following notebooks:
    * ```Main_notebook.ipynb``` - it contains code for:

        1. Analyzing MATLAB structure of Innopolis Motor dataset and convertiong experimental trials to .csv files. (Chapters 1 - 4).
        2. Using MNE to analyze and filter data. Chapter 5.
        3. Unzipping and analyzing BCI_competition_IV dataset. Chapter 6.
        4. LSTM_CNN machine learning pipeline. Chapter 7.
    * ```pipeline_motoral_Chrononet.ipynb``` - this notebooks contains code for analyzing Innopolis Motor dataset and building machine learning pipeline for ChronoNet model.
    * ```pipeline_visual.ipynb``` - this notebooks contains code for analyzing Innopolis Visual Necker Cube and building machine learning pipeline for LSTM model.
    * ```Thesys_wavelets.ipynb``` - this notebooks contains code for analyzing BCI_competition_IV dataset and generating spectrogram images using wavelet transform.
* ```Wavelets``` - this folder contains scripts for CNN machine learning pipeline for classification of spectrogram images generated from EEG data by wavelet transform.
* ```visual_LSTM/ChronoNet.py``` - scripts for machine learning pipelines for classification of EEG data from Innopolis Visual Necker Cube dataset.
* ```to_csv.m``` - script for generating .csv files for each trial from Innopolis Visual Necker Cube dataset. 
* ```utils``` - folder, containing scripts, used by ```visual_LSTM/ChronoNet.py```.
* ```motoral_ChronoNet/CNN_LSTM/LSTM_CNN/LSTM.py``` - scripts for machine learning pipelines for classification of EEG data from Innopolis Motor dataset.
* ```utils_motoric``` - folder, containing scripts, used by ```motoral_ChronoNet/CNN_LSTM/LSTM_CNN/LSTM.py```.

## Dependencies (Python):

```pip3 install -r requirements.txt```
