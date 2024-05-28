# 2024Hanium_code

#pre-trained only cup, bottle detect 

from typing import List, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import String
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from ultralytics import YOLO
import cv2
import math
import time

class yolov8Publisher(Node):
    def __init__(self):
        super().__init__('yolov8_publisher')
        qos_profile = QoSProfile(depth=10)
        self.publisher = self.create_publisher(String, 'yolov8_detection',qos_profile)
        self.bridge = CvBridge()
        #self.rate = self.create_rate(1)  #메시지 발행 속도 조절
    
    def publish_detection_msg(self, class_name):
        msg = String()
        msg.data = class_name
        self.publisher.publish(msg)
        self.get_logger().info(f"Published detection result: {class_name}")
        #self.rate.sleep()
        
def main(args=None):
    rclpy.init(args = args)
    yolov8_publisher = yolov8Publisher()
    last_time = time.time()
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    model = YOLO("yolov8n.pt")
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
    
    cup = classNames.index("cup")
    bottle = classNames.index("bottle")

    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        
        for r in results:
            boxes = r.boxes
        
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  #탐지된 f객체의 좌표
                #정수형으로 변환
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)              
                
            
                #탐지된 객체의 확률
               # confidence = math.ceil((box.conf[0]*100))/100
               # print("Confidence = ", confidence)
            
                #탐지된 객체의 클래스 인덱스
                cls = int(box.cls[0])
                
                if cls == cup or cls == bottle:
                    #탐지된 객체를 사각형으로 표시
                    cv2.rectangle(img, (x1,y1),(x2,y2), (255, 0, 255), 2)
                    #클래스 이름 출력
                    print("Class name = ", classNames[cls])
                    #클래스 이름 화면에 표시
                    cv2.putText(img, classNames[cls], [x1,y1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)
    
                    yolov8_publisher.publish_detection_msg(classNames[cls]) #클래스 이름을 다른 노드로 퍼블리쉬 
                
        cv2.imshow('Wdbcam', img)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    rclpy.spin(yolov8_publisher)
    yolov8_publisher.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()

    
