
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

The images are extracted from videos in Youtube platform and real-time videos. The tool used for labelling the text in images for training is [makesense.ai](https://www.makesense.ai/). Some of the images samples accquired from pre-recorded real-time video and graphic text videos are displayed below.

<p float="left">
   <img src="https://github.com/Apoorva-Udupa/Text-localization-and-Script-Identification-using-YOLOv5/blob/main/doc_images/image003.png" width="400" >&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="https://github.com/Apoorva-Udupa/Text-localization-and-Script-Identification-using-YOLOv5/blob/main/doc_images/image001.png" width="500" >
</p>




