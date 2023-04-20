import os
from pydub import AudioSegment, silence
from mutagen.wave import WAVE
import pandas as pd
import numpy as np
import joblib

def preprocess(audio_file):
    # Conversion to WAVE format
    out_file = os.path.splitext(audio_file)[0] + ".wav"
    if audio_file.endswith(".mp3"): 
        sound = AudioSegment.from_mp3(audio_file)
    elif audio_file.endswith(".ogg"):
        sound = AudioSegment.from_ogg(audio_file)
    sound.export(out_file, format="wav")
    return out_file

def get_conf_pred(audio_file):
    out_file=preprocess(audio_file)
    audio = WAVE(out_file)
    audio_info = audio.info
    total_duration = audio_info.length

    audio = AudioSegment.from_wav(out_file)
    dBFS = audio.dBFS
    silent_sections = silence.detect_silence(audio, min_silence_len=500, silence_thresh=dBFS-16)
    num_pauses=0
    total_duration_of_pauses = 0

    for start, stop in silent_sections:
        num_pauses+=1
        total_duration_of_pauses+=(stop/1000-start/1000)

    feature_set = {'total_duration': [],
            'num_pauses' : [],
            'total_silence' : []}
    feature_set['total_duration'].append(round(total_duration,2))
    feature_set['num_pauses'].append(num_pauses)
    feature_set['total_silence'].append(round(total_duration_of_pauses,2))
    info = pd.DataFrame(feature_set)
    os.unlink(out_file)

    # Predict using SVM Classifier
    svm_model = joblib.load('C:\\FinalYearProject\\AudioAnalysis\\svm_model.sav')
    pred = svm_model.predict(info)
    pred_acc = svm_model.predict_proba(info)
    
    return pred, pred_acc

""" audio_file = input("Enter path to audio file : ")
curr_path = "C:\\FinalYearProject\\AudioAnalysis\\"
pred, pred_acc = get_conf_pred(curr_path+audio_file)
print("Confidence Prediction : " + str(pred[0]))
print("Prediction Accuracy : %.2f" % round(max(pred_acc[0])*100, 2) + " %") """