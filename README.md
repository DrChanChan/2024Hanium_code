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

    
####################################################################################################################
# Oak_d_lite

import cv2
import depthai as dai
import numpy as np
y = 640
x = 400
def create_pipeline():

    pipeline = dai.Pipeline() #DepthAI 파이프라인을 초기화

    camRgb = pipeline.create(dai.node.ColorCamera)  #컬러 카메라를 파이프라인에 추가합니다. 이 노드는 컬러 이미지를 제공합니다.
    stereoDepth = pipeline.create(dai.node.StereoDepth) # 스테레오 심도 노드를 파이프라인에 추가합니다. 이 노드는 깊이 정보를 계산합니다.
    monoLeft = pipeline.create(dai.node.MonoCamera) # 왼쪽 모노 카메라를 파이프라인에 추가합니다. 이 노드는 왼쪽 카메라에서 모노 이미지를 제공
    monoRight = pipeline.create(dai.node.MonoCamera) #오른쪽 모노 카메라를 파이프라인에 추가합니다. 이 노드는 오른쪽 카메라에서 모노 이미지를 제공

    """
    camRgb 하이퍼파라미터 
    이미지 크기 = 640, 400
    카메라 보드 소켓 = CAM_A는 보드의 첫 번째 카메라 소켓을 나타냅니다.
    카메라 해상도 = 1080
    데이터가 순차적인지 여부를 설정합니다. False로 설정되어 있으므로, 데이터가 교대로 나오지 않음(False)되어 있다는 것을 의미
    """
    camRgb.setPreviewSize(y, x)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    
    """
    monoLeft, monoRight 하이퍼파라미터
    왼쪽 카메라 크기 = 400
    오른쪽 카메라 크기 = 400
    LEFT, RIGHT = 카메라 소켓 설정
    """
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    
    """
    stereoDepth 하이퍼파라미터
    신뢰 임계값을 설정합니다. 이 값은 깊이 맵에서 유효한 깊이 정보로 간주되는 임계값을 나타냅니다. = 190
    좌우 검사를 활성화합니다. 이는 깊이 맵을 생성할 때 왼쪽과 오른쪽 이미지 간의 일치 여부를 확인하는 데 사용됩니다. = True
    좌우 검사 임계값을 설정합니다. 이 값은 좌우 이미지의 일치 정도를 결정하는 데 사용됩니다. = 10
    양방향 필터의 시그마 값을 설정합니다. 이 필터는 이미지를 부드럽게 만들면서도 에지를 보존하는 데 사용됩니다. = 5
    메디안 필터를 설정합니다. 이 필터는 잡음을 제거하고 깊이 맵을 부드럽게 만드는 데 사용됩니다. = 7*7
    깊이 정렬을 설정합니다. 여기서는 왼쪽 이미지와 맞추도록 설정되어 있습니다. = LEFT
    확장된 disparity를 사용하지 않도록 설정 = False
    서브픽셀 정확도를 활성화합니다. 이는 깊이를 더 정확하게 계산하기 위해 서브픽셀 수준의 정확도를 제공합니다. = True
    """
    stereoDepth.initialConfig.setConfidenceThreshold(190)
    stereoDepth.setLeftRightCheck(True)
    stereoDepth.initialConfig.setLeftRightCheckThreshold(10)
    stereoDepth.initialConfig.setBilateralFilterSigma(5)
    stereoDepth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereoDepth.setDepthAlign(dai.CameraBoardSocket.LEFT) # default: Right
    stereoDepth.setExtendedDisparity(False)
    stereoDepth.setSubpixel(True)
    
    """
    후처리 설정
    """
    # Depth config
    depth_config = stereoDepth.initialConfig.get()
    depth_config.postProcessing.spatialFilter.enable = True
    depth_config.postProcessing.spatialFilter.holeFillingRadius = 2
    depth_config.postProcessing.spatialFilter.numIterations = 1
    depth_config.postProcessing.spatialFilter.alpha = 0.5
    depth_config.postProcessing.spatialFilter.delta = 20
    depth_config.postProcessing.temporalFilter.enable = True
    depth_config.postProcessing.temporalFilter.alpha = 0.4
    depth_config.postProcessing.temporalFilter.delta = 20
    depth_config.postProcessing.temporalFilter.persistencyMode = dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_4
    # depth_config.postProcessing.temporalFilter.persistencyMode = dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_8_OUT_OF_8
    depth_config.postProcessing.speckleFilter.enable = True
    depth_config.postProcessing.speckleFilter.speckleRange = 200
    stereoDepth.initialConfig.set(depth_config)

    """
    왼쪽 모노 카메라의 출력을 stereoDepth 노드의 왼쪽 입력에 연결
    오른쪽 모노 카메라의 출력을 stereoDepth 노드의 오른쪽 입력에 연결
    """
    monoLeft.out.link(stereoDepth.left)
    monoRight.out.link(stereoDepth.right)
    
    """
    여기서는 출력을 생성하여 파이프라인에서 결과를 외부로 보냅니다.
    """
    xoutRgb = pipeline.create(dai.node.XLinkOut)  # XLinkOut 노드를 생성, 파이프라인에서 데이터를 가져와 외부로 보냅니다.
    xoutDepth = pipeline.create(dai.node.XLinkOut) #  XLinkOut 노드를 생성,  깊이 맵 데이터를 외부로 보냅니다.   
    xoutRgb.setStreamName("rgb")  # XLinkOut 노드에 대한 스트림 이름을 설정합니다. 여기서는 RGB 이미지 스트림의 이름을 "rgb"로 지정
    xoutDepth.setStreamName("depth") # XLinkOut 노드에 대한 스트림 이름을 설정합니다. 여기서는 깊이 맵 스트림의 이름을 "depth"로 지정
    
    camRgb.preview.link(xoutRgb.input) #카메라에서 나온 프리뷰 데이터를 XLinkOut 노드의 입력에 연결합니다. 이를 통해 RGB 이미지 스트림이 외부로 전달
    stereoDepth.depth.link(xoutDepth.input)  #스테레오 깊이 노드에서 나온 깊이 데이터를 XLinkOut 노드의 입력에 연결합니다. 이를 통해 깊이 맵 스트림이 외부로 전달

    return pipeline

def get_distance_at_point(depth_frame, x, y):
    # 특정 좌표에 해당하는 깊이 값을 가져옵니다.
    distance = depth_frame[y, x]
    return distance

if __name__ == "__main__":
    deviceInfoVec = dai.Device.getAllAvailableDevices()  #모든 사용 가능한 DepthAI 장치의 정보를 가져옵니다.
    usbSpeed = dai.UsbSpeed.SUPER                        #USB 속도를 SUPER로 설정
    openVinoVersion = dai.OpenVINO.Version.VERSION_2021_4  #OpenVINO 버전을 2021년 4월 버전으로 설정합니다. 
    #OpenVINO은 Intel의 머신러닝 및 컴퓨터 비전 애플리케이션을 위한 툴킷

    #컬러 데이터, 깊이 데이터, 장치 정보를 저장하기 위한 빈 리스트 생성
    qRgbMap = {}  
    qDepthMap = {}
    devices = []



    for deviceInfo in deviceInfoVec:  #사용 가능한 모든 장치에 반복  
        device = dai.Device(openVinoVersion, deviceInfo, usbSpeed)  #depthAI 장치 초기화
        devices.append(device)   #초기화된 장치를 devices리스트에 추가
        print("===Connected to", deviceInfo.getMxId())  #현재 연결된 장치의 ID출력
        
        mxId = device.getMxId()   #일련 번호
        cameras = device.getConnectedCameras()  #카메라 수
        usbSpeed = device.getUsbSpeed()  #USB속도
        eepromData = device.readCalibration2().getEepromData() #제품의 정보를 가져옴
        
        print("   >>> MXID:", mxId)
        print("   >>> Num of cameras:", len(cameras))
        print("   >>> USB speed:", usbSpeed)
        
        if eepromData.boardName != "":
            print("   >>> Board name:", eepromData.boardName)  #보드 이름 출력
        if eepromData.productName != "":
            print("   >>> Product name:", eepromData.productName) #제품 이름 출력
            
   
        pipeline = create_pipeline()    ######## pipeline 변수에 create_pipeline()함수 저장
        device.startPipeline(pipeline)  # 생성된 파이프라인을 사용하여 DepthAI 장치에서 파이프라인을 시작합니다.
        
        
        """
        DepthAI 장치에서 RGB 및 깊이 데이터를 받아들이기 위한 출력 큐를 설정하고 매핑
        
        qRgb = "rgb" 스트림에 대한 출력 큐를 가져옵니다. 큐의 최대 크기는 4이며, 비차단 모드로 설정
        qDepth = "depth" 스트림에 대한 출력 큐를 가져옵니다. 큐의 최대 크기는 4이며, 비차단 모드로 설정
        streamName = RGB 데이터 스트림의 고유한 이름을 생성합니다. 제품 이름과 장치 ID를 결합하여 사용
        streamName2 = 깊이 데이터 스트림의 고유한 이름을 생성합니다. 제품 이름과 장치 ID를 결합하여 사용
        
        qRgbMap[streamName] =  RGB 데이터 스트림의 이름과 해당 큐를 매핑하여 딕셔너리에 저장합니다.
        qDepthMap[streamName2] = 깊이 데이터 스트림의 이름과 해당 큐를 매핑하여 딕셔너리에 저장합니다.
        """

        qRgb = device.getOutputQueue("rgb", 4, False) 
        qDepth = device.getOutputQueue("depth", 4, False)
        streamName = "rgb-" + eepromData.productName + mxId
        streamName2 = "depth-" + eepromData.productName + mxId
        qRgbMap[streamName] = qRgb
        qDepthMap[streamName2] = qDepth
        
    while True:
        for streamName, qRgb in qRgbMap.items():
            inRgb = qRgb.tryGet()
            if inRgb is not None:
                cv2.imshow(streamName, inRgb.getCvFrame())
        
        for streamName2, qDepth in qDepthMap.items():
            inDepth = qDepth.tryGet()
            if inDepth is not None:
                
                depthImgFloat = inDepth.getCvFrame()  # OpenCV 프레임으로 변환된 깊이 이미지를 가져옵니다.
                #formattedDepth = np.array2string(depthImgFloat, precision=3, suppress_small=True)
                #print(formattedDepth)

                min_, max_, _, _ = cv2.minMaxLoc(depthImgFloat)  # 깊이 이미지에서 최소값과 최대값을 계산합니다.
                #print(min_)
                #print(max_)
                depthImg = (depthImgFloat - min_) / (max_ - min_) #깊이 이미지를 0에서 255로 스케일링합니다.
                #depthImg = cv2.convertScaleAbs(depthImgFloat)  #스케일링된 깊이 이미지를 8비트로 변환합니다.
                #깊이 이미지에 색상을 적용하여 컬러 이미지를 생성합니다.
                coloredDepthImg = cv2.applyColorMap((depthImg * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                
                h, w = depthImgFloat.shape
                center_x = w // 2
                center_y = h // 2
                center_distance = depthImgFloat[center_y, center_x]
                
                # 소수점 3자리까지 출력
                print(f"Distance at the center ({center_x}, {center_y}): {center_distance:.3f} mm")
                cv2.circle(coloredDepthImg, (center_x, center_y), 5, (0, 0, 255), -1)

                cv2.imshow(streamName2, coloredDepthImg)

        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break


####################################
import cv2
import serial

def main():

    ############### PARAM ##############
    cap = cv2.VideoCapture(0)
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

    circles = [
        {"center": (180, 550), "radius": 20}, 
        {"center": (130, 550), "radius": 20},
        {"center": (80, 550), "radius": 20},
        {"center": (30, 550), "radius": 20},
    ]

    one_circle = 120
    two_circle = 90
    three_circle = 75
    four_circle = 50

    flip = False
    ####################################

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Get the initial frame dimensions
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Could not read frame.")
        return
    
    frame_height, frame_width, _ = frame.shape

    # Set the desired window size
    window_width, window_height = 1024, 600

    # Calculate the scaling ratio to fit the frame within the window size
    scale = min(window_width / frame_width, window_height / frame_height)

    # Calculate the scaled dimensions
    display_width = int(frame_width * scale)
    display_height = int(frame_height * scale)

    # Create a window without any decorations
    cv2.namedWindow('Rear Camera with Guidelines', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Rear Camera with Guidelines', display_width, display_height)
    cv2.setWindowProperty('Rear Camera with Guidelines', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    distance = 0.0
    # Start capturing video and display it
    while True:
        ret, frame = cap.read()

        if ser.in_waiting > 0:
            distance = ser.readline().decode('utf-8', errors='replace').rstrip()
            distance = int(float(distance))

        if not ret or frame is None:
            print("Error: Could not read frame.")
            break

        if flip == True:
            frame = cv2.flip(frame, 0)

        # Resize the frame to fit within the window size
        frame_resized = cv2.resize(frame, (display_width, display_height))

        cv2.putText(frame_resized, str(distance), (300,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 1)
        print(distance)

        if distance is not None:
            fill_count = 0

            if distance < four_circle:
                fill_count = 4
            elif distance < three_circle:
                fill_count = 3
            elif distance < two_circle:
                fill_count = 2
            elif distance < one_circle:
                fill_count = 1

        
        for i, circle in enumerate(circles):
            if i < fill_count:
                cv2.circle(frame_resized, circle["center"], circle["radius"], (0,0,255), -1)
            else:
                cv2.circle(frame_resized, circle["center"], circle["radius"], (0,0,255), 2)

        # Display the frame with guidelines
        cv2.imshow('Rear Camera with Guidelines', frame_resized)

        # 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ser.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

