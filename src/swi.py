#!/usr/bin/python3

#! @author Vsevolod (Seva) Ivanov

from picamera.exc import PiCameraValueError
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
from enum import Enum
from random import choice

class Swi:

    class Swipe(Enum):
        LEFT = -1
        RIGHT = 1

    def __init__(self, framerate=10, screen_w=480, screen_h=320,
                       cascade_classifier='res/haarcascade_frontalface_default.xml'
    ):
        # screen
        self.screen_w = screen_w
        self.screen_h = screen_h
        # raspberry pi camera
        self.camera = PiCamera()
        self.camera.resolution = (screen_w, screen_h)
        self.camera.framerate = framerate
        self.camera.exposure_mode = 'auto'
        cv2.namedWindow('window', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # machine learning classifer
        self.face_cascade = cv2.CascadeClassifier(cascade_classifier)

    def __del__(self):
        cv2.destroyAllWindows()

    def run(self):
        rawCapture = PiRGBArray(self.camera, size=(self.screen_w, self.screen_h))

        for frame in self.camera.capture_continuous(rawCapture, format='rgb', use_video_port=True):
            image = frame.array
            image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = self.detect_faces(image_bw)
            if (len(faces) > 0):
                faces = self.remove_far_faces(faces)
                if (len(faces) == 0):
                    continue
                x, y, w, h = self.find_closest_face(faces)
                cv2.rectangle(image_bw, (x, y), (x + w, y + h), (255, 255, 255), 2)
                print('I see ' + 'x'.join([str(w), str(h)]) + ' of you')
                self.swipe(image_bw, choice(
                    [self.Swipe.LEFT.value, self.Swipe.RIGHT.value]
                ))

            cv2.imshow('window', image_bw)
            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)

            if key == ord('q'):
                break

    def start_preview(self):
        self.camera.start_preview(window=(0, 0, self.screen_w, self.screen_h))
    
    def stop_preview(self):
        self.camera.stop_preview()

    def detect_faces(self, image):
        return self.face_cascade.detectMultiScale(image, 1.3, 5)

    def find_closest_face(self, faces):
        largest = 0
        largest_i = 0
        i = 0
        for (x, y, w, h) in faces:
            area = w * h
            if (area > largest):
                largest = area
                largest_i = i
            i+=1
        return faces[largest_i]

    def remove_far_faces(self, faces):
        close_faces = list()
        for (x, y, w, h) in faces:
            if ((w >= 150) or (h >= 150)):
                close_faces.append([x, y, w, h])
        return close_faces

    def swipe(self, image, direction, delay=1):
        rows, cols = image.shape
        step = 1
        steps = int(self.screen_w/step)
        for i in range(steps):
            M = np.float32([[1,0,i*step*direction],[0,1,0]])
            location = cv2.warpAffine(image,M,(cols,rows))
            cv2.imshow('window', location)
            cv2.waitKey(delay)

if __name__ == '__main__':
    swi = Swi()
    while True:
        try:
            swi.run()
        except picamera.exc.PiCameraValueError:
            pass
    del swi

