link tai dinh dang du lieu cho faster rcnn:

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="5nCch7JYrkKq5WvAgciq")
project = rf.workspace("tivasolutions-atf4b").project("taco-hgp6h")
version = project.version(1)
dataset = version.download("tfrecord")
                
Link tai dinh dang du lieu cho yolov11:

!pip install roboflow


from roboflow import Roboflow
rf = Roboflow(api_key="5nCch7JYrkKq5WvAgciq")
project = rf.workspace("tivasolutions-atf4b").project("taco-hgp6h")
version = project.version(1)
dataset = version.download("yolov11")