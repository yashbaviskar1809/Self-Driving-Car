from posture_detection import *

params, model_params = config_reader()
head_position= process('./sample_images/H1.jpg', params, model_params)