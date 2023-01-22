# MHPPS
## Multi-Modal Human Personality Profiling System
Predicts a user's personality from a combination of his/her tweets, a short audio interview and a video recording of his/her daily activities.
#
The system consists of 3 modules - 
  1. Tweet Analysis Module : Employs a machine learning model to predict the extroversion of a user with the help of the MBTI model.
  2. Audio Interview Analysis Module : Predicts the level of confidence from a short interview recording and assigns an appropriate extroversion score.
  3. Video Analysis Module : Determines how extroverted the user is from the number of persons he/she interacts with in a predefined time interval.  
#
The results of the above three modules are then combined using a weighted average/majority voting scheme to classify a person as an introvert/extrovert.
