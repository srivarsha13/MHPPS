from fer import FER
from fer import Video
import os
import sys

def get_face_expr(video_file):
    video = Video(video_file)

    emotion_detector = FER(mtcnn=True)
    raw_data = video.analyze(emotion_detector, display=False)

    df = video.to_pandas(raw_data)
    df = video.get_first_face(df)
    df = video.get_emotions(df)

    expr = df.mean(axis=0).idxmax()
    conf_level = df.mean(axis=0).max()*100

    return expr, conf_level

video_file = input("Enter path to video file : ")
expr, conf_level = get_face_expr(video_file)
print("Facial Expression : "+ expr.title())
print("Confidence Level : {:0.2f} %".format(conf_level))