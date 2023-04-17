# Ultralytics YOLO 🚀, GPL-3.0 license
'''
一般python只要是文件夹中就都会带一个__init__.py 但不强求
这个文件可以说是程序的入口 在外部导入这个包（也就是这个文件夹）的时候 一定要这个文件才能import 
'''
from .predict import DetectionPredictor, predict
from .train import DetectionTrainer, train
from .val import DetectionValidator, val
