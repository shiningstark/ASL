# ASL

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A HCI project which aims to help parents teach DHH(deaf and hearing hard) children American Sign Language. Our project is trying to create a prototype based on paper <<Augmenting Communication Between Hearing Parents and Deaf Children>>. A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).


---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run ASL_app.py.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python ASL_app.py
I set up video file path in ASL_app.py and model path in yolo.py. Others can change it for different usage.
```

### Usage
1. Input video: For our project, it should be used for real-time video processing so there will be a camera to take video as input. If you want to use local video to test, just change code in ASL_app.py. If self.video_path is 0, it uses webcam. If it is 1, it uses extenal camera. If it is local path, it plays local video. When our system work, the video will automatically play until closing window or video is over.

![image](https://github.com/shiningstark/ASL/blob/master/font/WechatIMG16.jpeg)

2. Mode: When users click 'START' button, it will change the mode. There are two mode: 1. object recognition 2 normal 
In object recognition mode, system will recognize all objects in the video and local them as image blow.

![image](https://github.com/shiningstark/ASL/blob/master/font/WechatIMG18.jpeg)

when user want to use normal mode, just click again 'START' button, and the log information will show what mode it is now.

3. Choose object: When user only want to recognize some special type object in video, click 'REFRESH' button and it will show a list in upper right where all the labels now in frame will show there.

![image](https://github.com/shiningstark/ASL/blob/master/font/WechatIMG17.jpeg)

And user can refresh the list by 'REFRESH' button.

After showing list, users can choose special label to show. Then the system will only local that type and show ASL video in main video and middle of right. The video is about how to express the label by ASL.

![image](https://github.com/shiningstark/ASL/blob/master/font/WechatIMG19.jpeg)

4. MultiGPU usage: It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```
    For windows system, you can download LabelImg to label your data and it will generate cooresponding xml file.

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with modification weights path when using ASL_app.py
    Remember to modify class path or anchor path.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---

## Some issues to know

1. The test environment is
    - Python 3.5.6
    - Keras 2.1.6
    - tensorflow-gpu 1.8.0

2. Default anchors are used. If you use your own anchors, probably some changes are needed.

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. If you want to train your own model and don't freeze any layers, you can use train.py. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.
