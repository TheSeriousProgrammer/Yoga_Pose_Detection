import cv2
import mediapipe as mp
from scipy.spatial.distance import cosine
from mediapipe.framework.formats.landmark_pb2 import Landmark
from os import listdir
from os.path import isfile
from random import sample
from typing import Iterable
import numpy as np



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
discardedJoints = [*range(0,11),*range(17,23),*range(29,33)] #Joints not to be considered for pose similarity


function = "euclidean"#"cosine"
vectorMode = "direct"#"differential"

def distanceBetweenVectors(vec1:np.array,vec2:np.array,axis:int):
    if function=="cosine":
        out = []
        assert((len(vec1.shape)-len(vec2.shape))<=1)
        for subvec in vec1 :
            out.append(cosine(subvec,vec2))
        return np.array(out)
    else:
        return np.linalg.norm(vec1-vec2,axis=axis)

def give_processed_vector(landmarks:Iterable[Landmark]):

    landmarks = [(landmark.x,landmark.y,landmark.z) for landmark in landmarks]
    
    if vectorMode == "differential" :
        pointsA = [11,11,11,13,13,14,15,23,23,23,24,24,25,25,26,27]
        pointsB = [12,13,15,15,23,16,16,24,25,27,26,28,26,27,28,28]

        pointsA:np.array = np.array([landmarks[x] for x in pointsA])
        pointsB:np.array = np.array([landmarks[x] for x in pointsB])

        differentialData = np.linalg.norm(pointsA-pointsB,axis=1)
        #print(differentialData)
        return differentialData
    else:
        importantPoints = np.array([landmarks[i] for i in range(len(landmarks)) if i not in discardedJoints]).flatten()
        return importantPoints


def convert_to_1_1_ratio(inpImage:np.array)->np.array :
    x , y ,channels = inpImage.shape
    top,bottom,left,right = 0,0,0,0
    if x==y :
        return inpImage
    elif(x>y):
        left = (x-y)//2
        right = (x-y)//2
    else:
        top = (y-x)//2
        bottom = (y-x)//2

    mod = cv2.copyMakeBorder(inpImage,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(0,0))

    return mod

aasanas = list(sorted(listdir("dataset/")))

sample_count=1

while sample_count<=4:

    confusionMatrix = np.array([[0,0,0,0,0,0,0,0]]*8)

    testImgs = []
    embeddings = []

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.0,min_tracking_confidence=0.0) as pose:

        for aasana in aasanas :

            directory = "dataset/"+aasana
            contents = list(sorted(listdir(directory)))
            selections = contents[:sample_count]
            otherSamples = contents[sample_count:]

            testImgs.append([ "dataset/"+aasana+"/"+sample for sample in otherSamples])

            for selection in selections:
                selectedImg = directory+"/"+selection
                image = cv2.imread(selectedImg)
                image = convert_to_1_1_ratio(image)

                result = pose.process(image)
                annotated_image = image.copy()

                mp_drawing.draw_landmarks(annotated_image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.imshow("checks",annotated_image)
                cv2.waitKey(200)
                embedding:np.array = give_processed_vector(result.pose_world_landmarks.ListFields()[0][1])
                embeddings.append(embedding)

        embeddings = np.array(embeddings)

        for i , aasana_samples in enumerate(testImgs) :
            for j , aasana_sample in enumerate(testImgs[i]):
                #print(testImgs[i][j])
                img = cv2.imread(testImgs[i][j])
                assert(img is not None)
                img = convert_to_1_1_ratio(img)
                result = pose.process(img)
                embedding:np.array = give_processed_vector(result.pose_world_landmarks.ListFields()[0][1])
                dist = distanceBetweenVectors(embeddings,embedding,1)
                detectedClass = np.argmin(dist)//sample_count
                confusionMatrix[i][detectedClass]+=1
        print(f"Confusion Matrix with {sample_count} samples")
        print(confusionMatrix)
        print(np.around(confusionMatrix/len(testImgs[0]),decimals=2))

    sample_count+=1
