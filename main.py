from imageai.Detection import VideoObjectDetection
import os
import cv2
from matplotlib import pyplot as plt
import torch
import simplepyble
from concurrent.futures import ThreadPoolExecutor
import speech_recognition as sr

torch.set_default_device("mps")
execution_path = os.getcwd()

camera = cv2.VideoCapture(0)

bt_adapter = simplepyble.Adapter.get_adapters()[0]

print(bt_adapter)

global bt_results
bt_results = []

global bt_string
bt_string = ""

resized = False

def bt_scan():
    
    bt_adapter.scan_for(1000)
    
    temp = bt_adapter.scan_get_results()
    global bt_results
    global bt_string
    for i in temp:
        if not (i.address() in bt_results):
            if not i.identifier() == "":
                bt_results.append(i.address())
                bt_string += i.identifier() + '    [' + i.address() + ']\n'

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        print(r.recognize_vosk(audio, language="english")) 


plt.rcParams['toolbar'] = 'None'
def forFrame(frame_number, output_array, output_count, returned_frame):
    plt.clf()
    plt.rcParams['toolbar'] = 'None'
    
    labels = []
    sizes = []

    counter = 0

    for eachItem in output_count:
        counter += 1
        labels.append(eachItem + " = " + str(output_count[eachItem]))
        sizes.append(output_count[eachItem])

    global resized

    if (resized == False):
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        resized = True

    plt.axis('off')
    plt.subplot(1, 1, 1)
    plt.tight_layout()
    
    bt_executor = ThreadPoolExecutor(max_workers=1)
    bt_executor.submit(bt_scan)
    
    # listen_executor = ThreadPoolExecutor(max_workers=1)
    # listen_executor.submit(listen)

    plt.text(100, 100, "good morning america")
    plt.text(100, 940, bt_string)
    plt.text(100, 960, "bluetooth devices")
    plt.imshow(returned_frame, interpolation="nearest", aspect='auto')

    plt.pause(0.01)
    

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolov3.pt"))
detector.loadModel()

plt.show()

detector.detectObjectsFromVideo(save_detected_video=False, return_detected_frame=True, camera_input=camera, per_frame_function=forFrame, frames_per_second=20, minimum_percentage_probability=30)

