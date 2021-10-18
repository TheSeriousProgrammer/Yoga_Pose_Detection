# Yoga Pose Classifier

This repository show cases pose identification can be achieved
with one shot learning applied over output embeddings of google
mediapipe library

This repo is more of a experiment than full blown project

Few shot learning / One shot learning for any task is generally
achieved with a base network trained to output similar vectors
for similar inputs and dissimilar vectors for dissimilar inputs

In this repo we tried to generallize the output vector from BlazePose
(a neural network used to predict and locate different joints in human body),
to make it output similar vectors for similar poses

These are the various joints predicted by BlazePose for a given person

![blazepose](https://raw.githubusercontent.com/TheSeriousProgrammer/Yoga_Pose_Detection/main/pose_tracking_full_body_landmarks.png)

These predicted vectors vary if the person is rotated inspite of the pose remaining the same, hence a processed vector, which gives relative positioning between joints is taken

The spacing between joints taken are highlighted below

![relative joints](https://raw.githubusercontent.com/TheSeriousProgrammer/Yoga_Pose_Detection/main/distance_btw_joints_considered.png)

This newly processed vector is immune to rotations, we can directly apply euclidian distance between reference samples and realtime feed to identify the concerned pose

Video Demo :

[Click To Watch Video Demo](https://raw.githubusercontent.com/TheSeriousProgrammer/Yoga_Pose_Detection/main/demo.mp4)