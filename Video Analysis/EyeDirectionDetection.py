import matlab.engine
import cv2
import os

def init_matlab_eng():
    engine = matlab.engine.start_matlab()
    return engine

def get_dir(video_file):
    curr_path = "C:\\FinalYearProject\\VideoAnalysis\\"
    engine = init_matlab_eng()
    cam = cv2.VideoCapture(curr_path+video_file)
    try :

        if not os.path.exists("output_edd"):
            os.makedirs("output_edd")

    except OSError:
        print("Error creating directory")

    #Initialize parameters
    frame_cur = 0
    frames_per_second = cam.get(cv2.CAP_PROP_FPS) 
    frames_captured = 0
    step = 1
    frame_count = 5

    dir_arr=[]

    while (True):
        ret, frame = cam.read()
        if ret:
            if frame_cur > (step*frames_per_second):  
                frame_cur = 0
                name = 'C:/FinalYearProject/VideoAnalysis/output_edd/fr' + str(frames_captured+1) + '.jpg'
                # print(name)
                cv2.imwrite(curr_path+name, frame)
                dir_arr+=[str(engine.get_eye_dir(name))]            
                frames_captured+=1
                if frames_captured>frame_count-1:
                    ret = False
            frame_cur += 1           
        if ret==False:
            break

    eye_dir = max(set(dir_arr), key = dir_arr.count)
    acc = dir_arr.count(eye_dir)/len(dir_arr)

    # Release space once done
    cam.release()
    cv2.destroyAllWindows()

    return eye_dir, acc

""" video_file = input("Enter path to video file : ")
curr_path = "C:\\FinalYearProject\\VideoAnalysis\\"
eye_dir, acc = get_eye_dir(curr_path+video_file)
aud_dir = input("Enter audience direction relative to speaker : ")
if(str(eye_dir).casefold() == aud_dir.casefold()) :
    print("Yes")
print("Eye Direction : " + eye_dir.title())
print("Prediction Accuracy : %.2f" % round(acc*100, 2) + " %") """