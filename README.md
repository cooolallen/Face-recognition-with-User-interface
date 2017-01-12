# Face recognition with User interface

In this project, we use [learning based model][6] to extract face features and to recognize who you are. Also, we create an elegant user interface in order to provide a good user experience. By interacting  with our UI, you can add/remove your face in our database.

Block Diagram
![Imgur](http://i.imgur.com/Urrf7Hh.png)


## Requirements
- Python 3.5.2
- [Tensorflow][1]
- [PyQt5][2]
- [OpenCV][3] 

## Pretrained-Model
- [FaceNet][4]

Please download FaceNet pretrained model from [here](https://drive.google.com/file/d/0B5MzpY9kBtDVSTgxX25ZQzNTMGc/view), extract and put it in CVFinal/pretrained_models/FaceNet/


## Usage
- <b>Run Face recognition</b> : 
```
python3 Top.py
```


- <b>Register identity face</b> :
	- Click the bounding box and type your name
	


- <b>Remove identity face</b> :
	- Choose the your name on the list and click remove buttom.

## Result
- Environment Setup
![Imgur](http://i.imgur.com/jeX7Obf.png)
- Register identity face
![Imgur](http://i.imgur.com/GyC0nBE.png)


## References
- [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks][5] Paper
- [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks][6] Code
- [FaceNet: A Unified Embedding for Face Recognition and Clustering][7] Paper
- [FaceNet: A Unified Embedding for Face Recognition and Clustering][6] Code

## License
MIT


[1]:https://www.tensorflow.org/
[2]:https://www.riverbankcomputing.com/software/pyqt/download5
[3]:http://opencv.org/
[4]:https://arxiv.org/abs/1503.03832
[5]:https://kpzhang93.github.io/MTCNN_face_detection_alignment/
[6]:https://github.com/davidsandberg/facenet
[7]:https://arxiv.org/abs/1503.03832

