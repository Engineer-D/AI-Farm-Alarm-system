# Raspberry Pi Weather and objection recognition device

# Project supervisor: Kayode Alade
# Project developer: Daramola David
# Date: 22/09/2021
# Time started: 12:04pm
# Time Ended:
#
# Project Description:
# This project device uses a Tensorflow Lite model to perform object detection on a live Pi camera feed.
# It recognises Human intrusion and animal invasion from the the live Pi camera feed.
# for better improvement in FPS, the webcam object runs in a separate thread from the main program.
# This device also takes environmental variables and send them to the cloud. Variables like:
#   * Temperature
#   * Humidity
#   * Soil Temperature
# Whenever there is an intrusion or invasion the pi sends notification to the user (SMS, Mail, Telegram message)
#

#_________________________________________ Code Session _______________________________________________________#

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import json #To convert our readings to json so to send to the API
import Adafruit_DHT # Library for the AM2301
from w1thermsensor import W1ThermSensor # For the ds18b20
import serial #for serial communication with GSM SIM800L

#______________________________________SIM800 Function Code Area____________________________________________________________#

print("Setting up Raspberry No Reply bot") # Intro text

#Speak with SIM800 -> gets AT command return as response
def SIM800(command):
    AT_command = command + "\r\n"
    ser.write(str(AT_command).encode('ascii'))
    time.sleep(1)
    if ser.inWaiting() > 0:
        echo = ser.readline() #waste the echo
        response_byte = ser.readline()
        response_str = response_byte.decode('ascii')
        return (response_str)
    else:
        return ("ERROR")

#Checks SIM800L status and connects with ShopifyAPI
def Init_GSM():
    if "OK" in SIM800("AT"):
        if ("OK" in (SIM800("AT+CMGF=1"))) and ("OK" in (SIM800("AT+CSMP=17,167,0,0"))):  # enble sms settings
            print("SIM800 Module -> Active and Ready")
    else:
        print("------->ERROR -> SIM800 Module not found")

#Receives the message and phone number and send that message to that phone number
def send_message(message, recipient):
    ser.write(b'AT+CMGS="' + recipient.encode() + b'"\r')
    time.sleep(0.5)
    ser.write(message.encode() + b"\r")
    time.sleep(0.5)
    ser.write(bytes([26]))
    time.sleep(0.5)
    print ("Message sent to customer")
    time.sleep(2)
    ser.flushInput()  # clear serial data in buffer if any

def Upload_and_send_sms(sendToServer, url, message, recipient):
    upload_to_server(sendToServer, url)
    time.sleep(3)
    print(SIM800("AT+CMGS=" + recipient))
    time.sleep(0.5)
    print(SIM800(message))
    ser.write(bytes([26]))
    time.sleep(0.5)
    print ("Message sent to customer")
    ser.flushInput()

def upload_to_server(sendToServer, url):    
    print('Preparing to send Readings')
    print(SIM800("AT+CIPSHUT"))
    time.sleep(0.5)
    print(SIM800("AT+SAPBR=1,1"))
    time.sleep(0.5)
    print(SIM800("AT+SAPBR=2,1"))
    time.sleep(0.5)
    print(SIM800("AT+SAPBR=3,1,\"APN\",\"web.gprs.mtnnigeria.net\"")) # APN JTM2M
    time.sleep(0.5)
    print(SIM800("AT+HTTPINIT"))
    time.sleep(0.5)
    print(SIM800("AT+HTTPPARA=CID,1"))
    time.sleep(0.5)
    print(SIM800("AT+HTTPPARA=URL," + url))
    time.sleep(0.5)
    print(SIM800("AT+HTTPPARA=\"CONTENT\",\"application/json\""))
    time.sleep(0.5)
    print(SIM800("AT+HTTPDATA=" + str(len(sendToServer)) + ",100000"))
    time.sleep(0.5)
    print(SIM800(sendToServer))
    time.sleep(0.5)
    print(SIM800("AT+HTTPACTION=1"))
    #check_server_response()
    time.sleep(0.5)
    print(SIM800("AT+HTTPREAD"))
    time.sleep(0.5)
    print(SIM800("AT+HTTPTERM"))
    time.sleep(0.5)
    print(SIM800("AT+SAPBR=0,1"))
    time.sleep(0.5)
    print(SIM800("AT+CIPSHUT"))
    time.sleep(0.5)
    print ("Readings Sent Online")
    time.sleep(3)
    ser.flushInput()  # clear serial data in buffer if any

def prepare_json(temp, hum, soil, message =None, recipient = None,intruder = None, upload =False):
    readings = [
        {
            "id": 1,
            "status": 0,
            "deviceID": "Pi_Cam",
            "farmId": 0,
            "frameID": 0,
            "tempEnvironment": temp,
            "humidity": hum,
            "objectDetected": intruder,
            "tempSoil": soil,
            "stateName": "Lagos",
            "stateCode": "NG-LA",
            "communityName": "Ajao Estate",
            "lgaName": "Oshodi/Isolo",
            "lgaCode": "100261",
            "hostAddress": "No 9 IK Peters, Ajao Estate Airport Rd",
            "hostName": "DeeKay",
            "longitude": "6.5546642400645965",
            "latitude": "3.3209537242912064",
            "batteryLevel": "Null",
            "signalStrength": "Null"
        }
    ]
    sendToServer = json.dumps(readings)
    url = "http://energy-api.milsat.africa/api/FarmStatus/UpdateFarmStatus"

    if upload:
        Upload_and_send_sms(sendToServer, url, message, recipient)
    else:
        upload_to_server(sendToServer, url)

cus_name = "DeeKay"
cus_phone = "08166700905"
smsSent = False

# COM defanition for windows -> Should change for Pi    
ser = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=15)  # timeout affects call duration and waiting for response currently 30sec
print("Established communication with", ser.name)
Init_GSM()
# ____________________________________________________________END of SIM800 Functions _________________________________________________________ #

def SendNotification():
    return('Human Detected, Sending SMS ...')

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

# Define Sensor variables
TempSensor =W1ThermSensor()
sensor = Adafruit_DHT.DHT22
pin = 23 #GPIO23

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    # Sensor Readings
    humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
    soil_temperature = TempSensor.get_temperature()

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            #cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window

            if label.split(":")[0].lower() == 'person' and smsSent == False:
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2) # added myself
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                print(SendNotification())
                intruder = label.split(":")[0].lower()
                smsSent = True
                text_message = "Hi " + cus_name + f". Your Farm has been invaded by {intruder}!!. -Engineer-D Innovations"
                #send_message(text_message, cus_phone)

                # Send sensor data
                if humidity is not None and temperature is not None and soil_temperature is not None:
                    print('Temp = {0:0.1f}*°C Humidity={1:0.1f}% Soil_Temp = {2:0.1f}*°C'.format(temperature, humidity, soil_temperature))
                    prepare_json(temperature, humidity, soil_temperature, text_message, cus_phone,intruder = intruder, upload=True)
                else:
                    print("Failed to get reading. Try again!")
                
            else:
                smsSent = False
                # Send Sensor data
                if humidity is not None and temperature is not None and soil_temperature is not None:
                    print('Temp = {0:0.1f}*°C Humidity={1:0.1f}% Soil_Temp = {2:0.1f}*°C'.format(temperature, humidity, soil_temperature))
                    prepare_json(temperature, humidity, soil_temperature, intruder = None)
                else:
                    print("Failed to get reading. Try again!")
                

    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()

# side bar
#  * To avoid slow processing don't let it draw boxes and scores around object of interests in each frame from
#       the webcam
#  * Find a way to process the location of animals or human detected in that exact same frame
#  * Could draw boxes for better visualization only when Human or animal is detected