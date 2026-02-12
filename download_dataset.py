from roboflow import Roboflow

rf = Roboflow(api_key="LISwWNR3pt6I8zjS1wBh")
project = rf.workspace("xrdaas").project("chest-xray-yolo")
version = project.version(4)
dataset = version.download("yolov8")
