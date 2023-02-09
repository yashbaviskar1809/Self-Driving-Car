import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes


from yolov3_tf2.deep_sort import preprocessing
from yolov3_tf2.deep_sort import nn_matching
from yolov3_tf2.deep_sort.detection import Detection
from yolov3_tf2.deep_sort.tracker import Tracker
from yolov3_tf2.tools import generate_detections as gdet
from PIL import Image
import _thread
import sys,os
from posture_detection import *

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', 'weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/A1_Trim.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', "Output//A1_Trim_out.mp4", 'output.mp4')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('height', 980, 'height')
flags.DEFINE_integer('width', 640, 'width')


def main(_argv):
    try:
        # Definition of the parameters
        max_cosine_distance = 0.5
        nn_budget = None
        nms_max_overlap = 1.0
        
        #initialize deep sort
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        if FLAGS.tiny:
            yolo = YoloV3Tiny(classes=FLAGS.num_classes)
        else:
            yolo = YoloV3(classes=FLAGS.num_classes)
        print(FLAGS.weights)
        yolo.load_weights(FLAGS.weights)
        logging.info('weights loaded')

        class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
        logging.info('classes loaded')

        try:
            print("Passing video: ",vid)
            vid = cv2.VideoCapture(int(FLAGS.video))
            print("Video name: ",vid)
        except:
            vid = cv2.VideoCapture(FLAGS.video)

        out = None

        if FLAGS.output:

            # by default VideoCapture returns float instead of int
            width = int(640)
            height = int(980)
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
            out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
            list_file = open('detection.txt', 'w')
            frame_index = -1 
        
        fps = 0.0
        count = 0
        prev_cord = 0
        while True:
            _, img = vid.read()

            img = cv2.resize(img, (FLAGS.width,FLAGS.height ))
            if img is None:
                logging.warning("Empty Frame")
                time.sleep(0.1)
                count+=1
                if count < 3:
                    continue
                else: 
                    break
            # print("Processing frame")
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            img_in = tf.expand_dims(img_in, 0)
            img_in = transform_images(img_in, FLAGS.size)

            t1 = time.time()
            boxes, scores, classes, nums = yolo.predict(img_in)
            classes = classes[0]
            names = []
            for i in range(len(classes)):
                names.append(class_names[int(classes[i])])
            names = np.array(names)
            converted_boxes = convert_boxes(img, boxes[0])
            features = encoder(img, converted_boxes)    
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features) if class_name in ["person","bicycle","motorbike","cat","dog","horse","sheep","cow","elephant","umbrella","bench"] ]
            
            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima suppresion
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]        

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                message = ""
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
                crop_img = img[int(bbox[0]): int(bbox[2]), int(bbox[1]): int(bbox[3])]
                params, model_params = config_reader()
                try:
                    # _thread.start_new_thread(process, (crop_img, params, model_params) )
                    if class_name =="person":
                        head_position = process(crop_img, params, model_params)
                        message += head_position + " "
                        print("person id: ", track.track_id, " headpose: ", head_position)
                except Exception as e:
                    print("Error: ",e)
                # print("person id: ", track.track_id," point: ",center)
                frame_center = FLAGS.width/3

                ##Crossing street left to right
                if center[0] < frame_center:
                    message += "LS"
                    if prev_cord == 0:
                        prev_cord = center[0]
                    else:
                        difference = center[0] - prev_cord
                        if difference>0:
                            print("moving to right person id : ",track.track_id )
                            message += "-->"
                        prev_cord = center[0]
                ##Crossing street right to left
                elif center[0] > frame_center*2:
                    message += "RS"
                    if prev_cord == 0:
                        prev_cord = center[0]
                    else:
                        difference  = center[0]-prev_cord
                        if difference<0:
                            print("moving to left person id: ",track.track_id)
                            message += "<--"
                        prev_cord = center[0]
                else:
                    # print("center")
                    print("Warning percen in center: ",track.track_id)
                    message += "Ctr"
                cv2.putText(img, class_name+' '+message + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                if "Ctr" in message:
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                else:
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

            ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
            for det in detections:
                bbox = det.to_tlbr() 
                # cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            
            # print fps on screen 
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                              cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            cv2.imshow('output', img)
            if FLAGS.output:
                out.write(img)
                frame_index = frame_index + 1
                list_file.write(str(frame_index)+' ')
                if len(converted_boxes) != 0:
                    for i in range(0,len(converted_boxes)):
                        list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
                list_file.write('\n')

            # press q to quit
            if cv2.waitKey(1) == ord('q'):
                break
        vid.release()
        if FLAGS.ouput:
            out.release()
            list_file.close()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)  

if __name__ == '__main__':
    try:
        app.run(main)
    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

