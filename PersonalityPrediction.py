from TweetAnalysis.MBTIClassification import get_extro_pred
from AudioAnalysis.ConfidenceEstimation import get_conf_pred
from VideoAnalysis.FacialExpressionRecognition import get_face_expr
from VideoAnalysis.EyeDirectionDetection import get_eye_dir

intro_text = """You are often the center of attention - and you like it that way. You aren't afraid to 
introduce yourselves to new people. You feel comfortable in large groups and rarely turn down
invitations for parties / gatherings. You tend to have a large no. of acquaintances and are always 
keen to expand your social circle.\n\n"""
extro_text = """You tend to be quiet, reserved, and introspective and prefer the company of only a 
few close friends. your idea of a good time is a quiet afternoon to yourself to enjoy your hobbies and 
interests that help you feel recharged and energized. You take plenty of time to think things out 
carefully before sharing your thoughts due to fear of disapproval or disagreement.\n\n"""
conf_text = """You possess an 'I can' attitude and enjoy nothing more than learning about the thoughts 
and feelings of people around you. You know you'll get what you want in due time, and don't run around
telling everyone about your grand plans. You do not seek approval from others; instead, all you focus on
is 'playing the game.'\n\n"""
nconf_text = """You feel hesitant to speak in front of unknown people and focus more on observing what
goes on around you. You fear that you will not be able to do a good job and will be judged by your peers.
You feel awkward in large gatherings and find it extremely difficult to maintain eye contact with people.
You tend to be overly insecure and are constantly scared of losing respect or relationships as a result 
of your actions.\n\n"""
vers_text = """You embrace changes in life and evaluate the situation rationally to determine the way
forward. You neither dwell on past experiences nor worry too much about the future; you focus rather
on dealing with the current issue at hand. You possess a high level of tolerance that enables you to
embrace change as a positive competitive advantage.\n\n"""
nvers_text = """You are not open to new ideas or changes and prefer adopting the bookish way to solve a
problem. You tend to set irrationally high and unattainable goals for yourselves and your team, causing
yourself to feel disheartened when they are not met. You find it difficult to look at things from the
viewpoint of others, leading to frequent conflicts with the people around you.\n\n"""
poise_text = """You are committed to speaking and acknowledging the truth without any fear of the
consequences. You present your authentic self at all times and tend to lay down the facts exactly as
they are rather than beating around the bush. You speak in a calm and composed manner and are not 
flustered by sudden, unxpected questions. You are not afraid to look people straight in the eye, and 
communicate in an even, unwavering tone.\n\n"""
npoise_text = """You have a hard time maintaining eye contact and concentrate your gaze on your 
surroundings. You focus too much on controlling your facial expressions and limit your gestures as much
as possible so as to not reveal anything. You feel stressed, and tend to stutter when posed with an 
unexpected question. You tend to distance yourselves from others, and are highly prone to depression 
and social withdrawal.\n\n"""
pred_summary = "Personality Summary :- \n\n"

username = input("Enter username : ")
extro_pred, extro_acc = get_extro_pred(username)
if extro_pred == "Extrovert" :
    pred_summary+=extro_text
else :
    pred_summary+=intro_text
print("Extroversion Prediction : " + extro_pred)
print("Prediction Accuracy : %.2f" % round(extro_acc*100, 2) + " %")
print()

audio_file = input("Enter path to audio file : ")
conf_pred, conf_acc = get_conf_pred(audio_file)
if str(conf_pred[0])=="Confident" :
    pred_summary+=conf_text
else :
    pred_summary+=nconf_text
print("Confidence Prediction : " + str(conf_pred[0]))
print("Prediction Accuracy : %.2f" % round(max(conf_acc[0])*100, 2) + " %")
print()

video_file = input("Enter path to video file : ")
fer_pred, fer_acc = get_face_expr(video_file)
if(fer_pred=="happy" or fer_pred=="neutral") :
    pred_summary+=vers_text
else :
    pred_summary+=nvers_text
print("Facial Expression : "+ fer_pred.title())
print("Prediction Accuracy : {:0.2f} %".format(fer_acc))
print()

edd_pred, edd_acc = get_eye_dir(video_file)
if(edd_pred=="Straight") :
    pred_summary+=poise_text
else :
    pred_summary+=npoise_text
print("Eye Direction : " + edd_pred.title())
print("Prediction Accuracy : %.2f" % round(edd_acc*100, 2) + " %")
print()

soc_score = 0.15*extro_acc + 0.35*(0.4*edd_acc + 0.6*fer_pred) + 0.50*conf_acc
print("Your Sociability Score : {:.2f} / 100\n\n".format(soc_score))
print(pred_summary)