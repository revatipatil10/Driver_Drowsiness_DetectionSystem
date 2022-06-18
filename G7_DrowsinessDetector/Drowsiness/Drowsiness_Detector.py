#GROUP-7 DRIVER DROWSINESS DETECTION

#Import necessary libraries
import tkinter
from tkinter import*
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import time
import dlib
import cv2
import playsound
from playsound import playsound
import PIL.Image, PIL.ImageTk



#This function calculates and return eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)
    return ear

def mouth_aspect_ratio(mouth): 
	A = distance.euclidean(mouth[13], mouth[19])
	B = distance.euclidean(mouth[14], mouth[18])
	C = distance.euclidean(mouth[15], mouth[17])
	D = distance.euclidean(mouth[12], mouth[16])

	MAR= (A+B+C)/(3*D)
	return MAR
            
def start():
    global canvas,imagecon
    #Minimum threshold of eye aspect ratio below which alarm is triggerd
    EYE_ASPECT_RATIO_THRESHOLD = 0.3

    #Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
    EYE_ASPECT_RATIO_CONSEC_FRAMES = 20
    MAR_ASPECT_RATIO_CONSEC_FRAMES = 20

    # Another constant which will work as a threshold for MAR value     
    MAR_THRESHOLD = 0.6

    #COunts no. of consecutuve frames below threshold value
    COUNTER = 0
    count_yawn = 0
    sleep_count = 0
    mar_counter = 0
    
    #Load face cascade which will be used to draw a rectangle around detected faces.
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


    #Load face detector and predictor, uses dlib shape predictor file
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    #Extract indexes of facial landmarks for the left and right eye
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

    

    #Start webcam video capture
    video_capture = cv2.VideoCapture(0)

    #Give some time for camera to initialize(not required)
    time.sleep(1)
    #ret, frame = video_capture.read()

    playsound('sound/warning.mp3')
    
    while(True):
        #Read each frame and flip it, and convert to grayscale
        canvas.delete("all")
        ret, frame = video_capture.read()
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
        #Detect facial points through detector function
        faces = detector(gray, 0)

        #Detect faces through haarcascade_frontalface_default.xml
        face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

        #Draw rectangle around each face detected
        for (x,y,w,h) in face_rectangle:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        
        #Detect facial points
        for face in faces:

            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            #Get array of coordinates of leftEye and rightEye
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            #print(mouth)
            #Calculate aspect ratio of both eyes
            leftEyeAspectRatio = eye_aspect_ratio(leftEye)
            rightEyeAspectRatio = eye_aspect_ratio(rightEye)

            
            eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2
            #print(eyeAspectRatio)  #/////////

            #Calculate aspect ratio of mouth
            MAR=mouth_aspect_ratio(mouth)

            #Use hull to remove convex contour discrepencies and draw eye shape around eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthEyeHull = cv2.convexHull(mouth)
            
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthEyeHull], -1, (0, 255, 0), 1)


            #Detect if eye aspect ratio is less than threshold
            if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
                COUNTER += 1
                
                #If no. of frames is greater than threshold frames,
                if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                    sleep_count += 1
                    playsound('sound/warning.mp3')
                    cv2.imwrite("frame/frame_sleep%d.jpg" % sleep_count, frame)
                    cv2.drawContours(frame, [leftEye], -1, (0, 0, 255), 1)
                    cv2.drawContours(frame, [rightEye], -1, (0, 0, 255), 1)
                    playsound("sound/alarm.mp3")
                    cv2.putText(frame, "DROWSINESS ALERT!!", (140,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    print("UnSafe, please WAKEUP !!")


            else:
                #cv2.putText(frame, "Safe Drive", (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                print("Safe, Enjoy Drive ")
                COUNTER = 0 

            if MAR > MAR_THRESHOLD:
                mar_counter += 1
                
                if mar_counter >= MAR_ASPECT_RATIO_CONSEC_FRAMES:
                    count_yawn += 1
                    cv2.imwrite("frame/frame_sleep%d.jpg" % count_yawn, frame)#if count_yawn >= MAR_ASPECT_RATIO_CONSEC_FRAMES:
                    cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1)
                    cv2.putText(frame, "DROWSINESS ALERT!!", (140, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print("Unsafe??.... ")
                    #playsound('sound/warning.mp3')

            '''else:
                    
                    COUNTER = 0
                    cv2.putText(frame, "Safe Drive", (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    print("Safe, Enjoy Drive ")'''

               
            
 
            #Print EAR value on screen
            cv2.putText(frame, "EAR: {:.2f}".format(eyeAspectRatio), (430, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "MAR: {:.2f}".format(MAR), (430, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Sleep Count: {:.2f}".format(sleep_count), (430, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            
            cv2.putText(frame,"Press 'q' to exit" ,(30, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
        #canvas.itemconfig(imagecon, image = photo)
        #canvas.pack()

            #cv2.putText(frame, "Safe Driver", (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('frame',frame)

       
        #Show video feed
        
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    video_capture.release()
    cv2.destroyAllWindows()

   
window = tkinter.Tk()

img=PhotoImage(file='logo2.png')
logo = Label(window,image=img).pack()



#root.mainloop()

window.title("DRIVER DROWSINESS DETECTION SYSTEM")
window.geometry("1500x1700")
#window['background']="#BFF7C5"
message = tkinter.Label(window, text="DRIVER DROWSINESS DETECTION SYSTEM",bg="#893BFF"  ,fg="white"  ,width=60  ,height=3,font=('times', 25, 'italic bold underline'))
message.place(x=100, y=160)       #100,20




#img=cv2.imread('12.jpg')
#RGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
canvas = tkinter.Canvas(window, width = 10, height = 10)  #500,480,bg="#BFF7C5"
canvas.place(x=100,y=175)
#photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(RGB))   
#imagecon=canvas.create_image(0, 0, image = photo, anchor = tkinter.NW)



#name= cv2.imread('car-insurance.png')
#RGB=cv2.cvtColor(name,cv2.COLOR_BGR2RGB)
#canvas1 = tkinter.Canvas(window, width = 500, height = 500)
#canvas1.place(x=450,y=200)   
#photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(RGB))
#canvas1.create_image(0, 0, image=photo, anchor=tkinter.NW)



# Button that lets the user take a snapshot
btn_snapshot=tkinter.Button(window , text="Video Start", width=20,height=2,bg="#07b8b4",fg="white",command=start,activebackground = "orange" ,font=('times', 15, ' bold '))
btn_snapshot.place(x=555,y=325)


btn_snapshot=tkinter.Button(window, text="Window EXIT", width=20,height=2,bg="#07b8b4",fg="white", command=window.destroy,activebackground = "orange" ,font=('times', 15, ' bold '))
btn_snapshot.place(x=555,y=475)

message = tkinter.Label(window,text="#Safe" ,bg="#F660AB"  ,fg="white"  ,width=8  ,height=2,font=('times', 30, 'italic bold'))
message.place(x=100, y=160)         
message = tkinter.Label(window,bg="blue"  ,fg="white"  ,width=5  ,height=1,font=('times', 30, 'italic bold '))
message.place(x=100, y=258)
message = tkinter.Label(window,text="#Drive" ,bg="#F660AB"  ,fg="white"  ,width=8  ,height=2,font=('times', 30, 'italic bold '))
message.place(x=100, y=310)
message = tkinter.Label(window,bg="blue"  ,fg="white"  ,width=5  ,height=1,font=('times', 30, 'italic bold '))
message.place(x=100, y=408)
message = tkinter.Label(window,text="#Save" ,bg="#F660AB"  ,fg="white"  ,width=8  ,height=2,font=('times', 30, 'italic bold '))
message.place(x=100, y=460)
message = tkinter.Label(window,bg="blue"  ,fg="white"  ,width=5  ,height=1,font=('times', 30, 'italic bold '))
message.place(x=100, y=558)
message = tkinter.Label(window,text="#Life" ,bg="#F660AB"  ,fg="white"  ,width=8  ,height=2,font=('times', 30, 'italic bold '))
message.place(x=100, y=610)


bar = tkinter.Label(window,bg="#800517"  ,fg="white"  ,width=30 ,height=1)
bar.place(x=300, y=270)
bar = tkinter.Label(window,bg="#800517"  ,fg="white"  ,width=30 ,height=1)
bar.place(x=300, y=420)
bar = tkinter.Label(window,bg="#800517"  ,fg="white"  ,width=30 ,height=1)
bar.place(x=300, y=570)
bar = tkinter.Label(window,bg="#00FF00"  ,fg="white"  ,width=25 ,height=1)
bar.place(x=400, y=350)
bar = tkinter.Label(window,bg="#00FF00"  ,fg="white"  ,width=25 ,height=1)
bar.place(x=400, y=500)

bar = tkinter.Label(window,bg="#800517"  ,fg="white"  ,width=30 ,height=1)
bar.place(x=840, y=270)
bar = tkinter.Label(window,bg="#800517"  ,fg="white"  ,width=30 ,height=1)
bar.place(x=840, y=420)
bar = tkinter.Label(window,bg="#800517"  ,fg="white"  ,width=30 ,height=1)
bar.place(x=840, y=570)
bar = tkinter.Label(window,bg="#00FF00"  ,fg="white"  ,width=25 ,height=1)
bar.place(x=780, y=350)
bar = tkinter.Label(window,bg="#00FF00"  ,fg="white"  ,width=25 ,height=1)
bar.place(x=780, y=500)

message = tkinter.Label(window,text="#Safe" ,bg="#F660AB"  ,fg="white"  ,width=8  ,height=2,font=('times', 30, 'italic bold'))
message.place(x=1066, y=160)         
message = tkinter.Label(window,bg="blue"  ,fg="white"  ,width=5  ,height=1,font=('times', 30, 'italic bold '))
message.place(x=1135, y=258)
message = tkinter.Label(window,text="#Drive" ,bg="#F660AB"  ,fg="white"  ,width=8  ,height=2,font=('times', 30, 'italic bold '))
message.place(x=1066, y=310)
message = tkinter.Label(window,bg="blue"  ,fg="white"  ,width=5  ,height=1,font=('times', 30, 'italic bold '))
message.place(x=1135, y=408)
message = tkinter.Label(window,text="#Save" ,bg="#F660AB"  ,fg="white"  ,width=8  ,height=2,font=('times', 30, 'italic bold '))
message.place(x=1066, y=460)
message = tkinter.Label(window,bg="blue"  ,fg="white"  ,width=5  ,height=1,font=('times', 30, 'italic bold '))
message.place(x=1135, y=558)
message = tkinter.Label(window,text="#Life" ,bg="#F660AB"  ,fg="white"  ,width=8  ,height=2,font=('times', 30, 'italic bold '))
message.place(x=1066, y=610)

#vertical green line
message = tkinter.Label(window,bg="#E6FF33"  ,fg="white"  ,width=1  ,height=2,font=('times', 30, 'italic bold'))
message.place(x=500, y=310)
message = tkinter.Label(window,bg="#E6FF33"  ,fg="white"  ,width=1  ,height=2,font=('times', 30, 'italic bold'))
message.place(x=500, y=460)
message = tkinter.Label(window,bg="#E6FF33"  ,fg="white"  ,width=1  ,height=2,font=('times', 30, 'italic bold'))
message.place(x=830, y=310)
message = tkinter.Label(window,bg="#E6FF33"  ,fg="white"  ,width=1  ,height=2,font=('times', 30, 'italic bold'))
message.place(x=830, y=460)


#bar = tkinter.Label(window,bg="orange"   ,width=20 ,height=1)
#bar.place(x=610, y=400)
#bar = tkinter.Label(window,bg="white"  ,width=20 ,height=1)
#bar.place(x=610, y=420)
#bar = tkinter.Label(window,bg="green" ,width=20 ,height=1)
#bar.place(x=610, y=440)

window.bind('j', start)
window.mainloop()
#Finally when video capture is over, release the video capture and destroyAllWindows

