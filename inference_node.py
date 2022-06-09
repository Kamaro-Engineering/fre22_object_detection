#!/usr/bin/env python3

#import tensorflow as tf
#tf.debugging.set_log_device_placement(True)
from sensor_msgs.msg import Image
#from cv_bridge import CvBridge
import numpy as np
import cv2
import rospy
from std_msgs.msg import Header
import onnxruntime

#bridge = CvBridge()

class InferenceNode:
        def __init__(self, model_path, frame_size):
                #self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.ort_session = onnxruntime.InferenceSession("./models/danelion-v3.onnx", providers=['CUDAExecutionProvider'])
                self.width, self.height = frame_size
                #self.interpreter.allocate_tensors()
                #self.input_index = self.interpreter.get_input_details()[0]["index"]
                #self.output_index = self.interpreter.get_output_details()[0]["index"]
                self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, callback=self.image_cb, queue_size=1)
                self.segmentation_pub = rospy.Publisher("/segmentation", Image, queue_size=1)

        def image_cb(self, msg):
                #frame = (bridge.imgmsg_to_cv2(msg) / 255).astype(np.float32)
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3)).astype(np.float32) / 255.0
                frame = cv2.resize(frame, (self.width, self.height))[np.newaxis,:,:,:]
                #self.interpreter.set_tensor(self.input_index, frame)
                #self.interpreter.invoke()
                #predictions = np.rint(self.interpreter.get_tensor(self.output_index) * 255).astype(np.uint8)
                predictions = (self.ort_session.run(None, {"conv2d_input": frame})[0] * 255).astype(np.uint8)
                predictions = cv2.resize(predictions[0,:,:,:], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                predictions_msg = Image() #bridge.cv2_to_imgmsg(predictions)
                predictions_msg.encoding = "rgb8"
                predictions_msg.header = msg.header
                predictions_msg.width = predictions.shape[1]
                predictions_msg.height = predictions.shape[0]
                predictions_msg.step = predictions.shape[1] * 3
                predictions_msg.data = predictions.tobytes()
                self.segmentation_pub.publish(predictions_msg)
                print("published", predictions.sum())
                # msg.header = Header()
                # self.segmentation_pub.publish(msg)

if __name__ == "__main__":
        rospy.init_node("inference_node")
        inference_node = InferenceNode(model_path="models/danelion-v2-f16.tflite", frame_size=(640, 480))
        rospy.spin()

