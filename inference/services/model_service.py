# In order to use `import cv2`, necessary libraries need to be loaded by following code before the importing.
import ctypes

import io
import cv2
import torch
import base64
import numpy as np
from PIL import Image
import onnxruntime as rt
from numpy import random

from inference.helper.model_utils import letterbox, non_max_suppression, drawBBox

class model:
    def __init__(self, 
                model_weights = './inference/helper/asserts/yolor_csp_x_star.quat.onnx', 
                imgsz = (1280, 1280), 
                threshold = 0.4,
                iou_thres = 0.6,
                names = 'inference/helper/asserts/coco.names'):
        '''
        Model config

        model_weights : weight of model ,default /src/assert/yolor_csp_x_star.qunt.onnx
        max_size : max size of image (widht, height) ,default 896
        names : name of class ref ,default coco/src/assert/coco.names
        '''

        self.names = self.load_classes(names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.imgsz = imgsz
        self.threshold = threshold
        self.iou_thres = iou_thres

        # load model
        sess_options = rt.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model = rt.InferenceSession(model_weights, sess_options)
        self.model.set_providers(['CPUExecutionProvider'])

        print("Load model done!!")

    def load_classes(self, path):
        '''
        Loading coco label

        path : path of coco label file
        '''

        with open(path, 'r') as f:
            names = f.read().split('\n')
        # filter removes empty strings (such as last line)
        return list(filter(None, names))

    def preProcessing(self, image_base64):
        '''
        Preprocessing image before feed to model from byte image to suitable image
        [byte image -> numpy image -> suitable numpy image]

        image_byte : byte image that upload from FastAPI
        '''

        ## Convert byte image to numpy array image
        in_memory = io.BytesIO(image_base64)
        image_pil = Image.open(in_memory)
        bgr_img = np.array(image_pil)

        ## Prepocessing image before feed to model
        # Padded resize
        inp = letterbox(bgr_img, new_shape=self.imgsz, auto_size=64)[0]
        # BGR to RGB
        inp = inp[:, :, ::-1].transpose(2, 0, 1)
        # Normalization from 0 - 255 (8bit) to 0.0 - 1.0
        inp = inp.astype('float32') / 255.0
        # Expand dimention to have batch size 1
        inp = np.expand_dims(inp, 0)

        return None, [bgr_img, inp]
    
    def detect(self, image):
        '''
        Object detection from coco label
        model name: YOLOR_CSP_X

        image : input image that already prepocessing 
        '''
        ort_inputs = {self.model.get_inputs()[0].name: image}
        pred = self.model.run(None, ort_inputs)[0]
        return None, pred

    def postProcessing(self, image_d, image_p, pred):
        '''
        After get result from model this function will post processing the result before
        seat a output
        '''

        # NMS
        with torch.no_grad():
            pred = non_max_suppression(torch.tensor(pred), conf_thres=self.threshold, iou_thres=self.iou_thres)
        det = pred[0]

        # Check have prediction
        if det is not None and len(det):
            # Rescale boxes from img_size to origin size
            _, _, height, width = image_p.shape
            h, w, _ = image_d.shape
            det[:, 0] *= w/width
            det[:, 1] *= h/height
            det[:, 2] *= w/width
            det[:, 3] *= h/height
            for x1, y1, x2, y2, conf, cls in det:
                # Draw BBox
                label = '%s %.2f' % (self.names[int(cls)], conf)
                image_d = drawBBox((x1, y1), (x2, y2), image_d, label, self.colors[int(cls)])

        # Convert to byte image
        _, im_buf_arr = cv2.imencode(".jpg", image_d)
        byte_im = base64.b64encode(im_buf_arr)

        return None, byte_im


    def inference(self, image_base64):
        log, result = self.preProcessing(image_base64)
        log, pred = self.detect(result[1])
        log, result = self.postProcessing(result[0], result[1], pred)

        return None, result