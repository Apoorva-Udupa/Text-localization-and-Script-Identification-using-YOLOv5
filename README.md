
# Text localization and script identification using Yolov5 

(in progress) A ML and Big Data Analytics project which detects the text present in images and natural scenes using YOLOv5

## Introduction
Script identification in color imagery and video content has gained popularity over the years in the field of research with an increase in demand for the extraction of textual information from natural scene images. In this paper, Text localization/ text detection and its corresponding script identification are performed on color images and videos. The texts are spotted in color images and videos and their scripts are distinguished in either English, Hindi, or Kannada language. For text detection and script recognition, the state-of-the-art method called YOLOv5 is used. YOLO is a real-time, single-shot object detection algorithm with better performance and higher accuracy than other object detection algorithms. The proposed model is trained with a custom dataset containing images and labels of all the text present in each and every image and the model is tested for different scenarios like different backgrounds, fonts, orientations, resolutions, and disturbances in the images to check its robustness. Finally, the proposed model is compared with other existing models/ algorithms for text detection and script identification in color imagery. 

## Challenges
Text detection and script identification in complex natural scenes can be very challenging. The text localization in color images has many complexities like different background colors in image, scene complexity, different fonts of text, different text orientations, light glares and lens flares in images, illumination effect in images etc. Some of the challenges found in text localization and script identification in color images and videos are: 

| Environment         | Image capture          | Text Content            |
| ------------------- | ---------------------- | ----------------------- |
| Scene intricacy     | Lens flare             | Different orientations  |
| Uneven illumination | light glares in images | Fonts variations        |
|       ---           | Resolutions/ burring   | Multi-lingual texts     |
|       ---           | Image distortion       | Text thickness          |

## Data Collection

There are many videos and natural scenes available having the tri-lingual scripts i.e. English, Hindi, and Kannada. But the standard database having the images of these tri-lingual scripts is scarcely found. For the proposed model, two sets of the database were constructed where one was used for training the model while the other database was used for validation. The images were collected from several sources. Most of the images with horizontally oriented scripts were acquired from translation videos and the natural scene images were obtained from the videos having street-view data of places located in Karnataka, Varanasi, Delhi, and Mumbai. The natural scene images have scripts of different orientations, colors, backgrounds, and also with varieties of image resolutions. The images with noise and disturbances were also used to train the model to make it robust and efficient. The training database has images having characters of Hindi, English, and Kannada as well as words from these scripts for better script identification purposes. 

<p align = 'center'>
   <img src="https://github.com/Apoorva-Udupa/Text-localization-and-Script-Identification-using-YOLOv5/blob/main/doc_images/image006.jpg" width="800" >
</p>


The images are extracted from videos in Youtube platform and real-time videos. The tool used for labelling the text in images for training is [makesense.ai](https://www.makesense.ai/). Some of the images samples accquired from pre-recorded real-time video and graphic text videos are displayed below.

<p float="left">
   <img src="https://github.com/Apoorva-Udupa/Text-localization-and-Script-Identification-using-YOLOv5/blob/main/doc_images/image003.png" width="400" >&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="https://github.com/Apoorva-Udupa/Text-localization-and-Script-Identification-using-YOLOv5/blob/main/doc_images/image001.png" width="500" >
</p>

## WorkFlow 

The model proposed for the project is YOLO. Yolo is a single-shot object detection algorithm as it processes an entire image once using a single CNN. YOLO is very fast in computation and can be used in a real-time environment. This algorithm has a very high accuracy range and has good performance with 45FPS. Basically, YOLO works by dividing the input image into grids of equal size (S x S) and then it predicts a class and a bounding box of objects present in the grid for every grid in the image. The input to the YOLO is images with corresponding labels (ground truth bounding boxes) and then YOLO will predict the bounding boxes with confidence scores and a class probability map for each image. Finally, the output obtained from YOLO on the test image contain final detections with bounding boxes around the detected objects in an image.
<p align = 'center'>
   <img src="https://github.com/Apoorva-Udupa/Text-localization-and-Script-Identification-using-YOLOv5/blob/main/doc_images/image005.jpg" width="800" >
</p>

**The parameters set to train the model are:**

No of GPUs: 01

No. of Classes: 3

Name of Classes: ‘Kannada’, ‘Hindi’, ‘English’

Batch size: 16

Epochs: 100

Weights: yolov5s.pt

After the successful construction of the dataset, this training dataset is given to the model to train it. It is seen that the more the images and labels are given to the model, the more precise it is able to predict and identify the scripts. The environment used to run the model is Google Colab using 1 GPU as the hardware accelerator. Packages like Torch OS and ipython are imported to display the images. The darknet53 is installed which is the backbone of YOLOv5 and constitutes of Convolution Neural Network with 125 layers. The other requirements are installed into google collab which creates yolov5 file directory.

A custom file in YAML format is created with information such as class labels and a number of classes and is uploaded into the data file in the yolov5 file present in the repository. The model is trained after specifying image dimensions which are set to 416, the batch size is 16 and the number of epochs is 100. The location of the actual dataset and YAML file is specified. Here, yolov5s weights, which is a small model, are chosen and the cache is specified to cache the images in the GPU. Weights and biases are downloaded and the Yolo model starts training the convolutional neural network. 

After setting the necessary parameters and the training of the model is initiated, each epoch run gives information like GPU memory, classes, number of labels, image size, and mAP values with a confidence score of a bounding box that is predicting the text in the images. After the epoch runs are completed, the last and the best weights are obtained. By validating the best weights, we get the model summary. Results of training are stored in a trained file in the repository. If the model does not give satisfactory results, batch size and number of epochs are changed and the model is trained again. The results are visualized with the tensorboard to get metrics and graphs which helps us to know if the model still needs to be trained.

Testing of the model is done by uploading the sample video or images into the colab repository. The detection script is set to run by specifying the sample video or images by choosing the best weights from the training process of 100 epochs. The confidence score is set over 0.2 for the correct prediction. The result obtained from the prediction script is stored in the detect file in the repository. The sample video/image is downloaded from the repository which contains the text which is localized and identified using the bounding box. The results obtained from the model are discussed in the below section.

## Video Results (model tested on sample video)

*Detection on graphic text video*

https://user-images.githubusercontent.com/88142527/164990516-493797ee-f1c8-4785-83b8-41a369b9607f.mp4


*Detection on pre-recorded real-time street video*

https://user-images.githubusercontent.com/88142527/164990598-cab6ba38-d3b2-4816-8038-f102376b8e52.mp4

## Experiment Results on Images 

To test the efficacy of the model, the trained model is tested on several graphic text translation videos and real-time recorded street view videos. The model has a speed of 45 frames per second and hence it is able to detect texts in the video very fast and also identifies the script of the localized text. The screenshots of the sample video and the Yolo detections on the video are displayed. The YOLO detects the text by drawing the bounding box around it and identifies the script as one among 3 classes and displays the confidence score for each detection.






