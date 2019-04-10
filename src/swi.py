#!/usr/bin/python3

#! @author Vsevolod (Seva) Ivanov

from picamera.exc import PiCameraValueError
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
from enum import Enum
from random import choice, randint

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
        # generate & encode poems into list of words
        poem = self.poem()
        print(poem)
        self.TEXT_BINARY = self.convert_text_binary(poem).split()
        self.TEXT_HUMANS = poem.split()

    def __del__(self):
        cv2.destroyAllWindows()

    def run(self):
        # controls
        draw_face_rectangles = True

        # loop variables
        frame_nb = 0
        face_offset = 40
        face_seen_at = -1
        face_seen_image = None
        wait_n_frames_swipe = 30
        text = self.TEXT_BINARY
        text_loc = (10, 10)
        max_inline_words = 6
        text_lines = list()
        text_frame_freq = 4
        text_for_humans = False

        rawCapture = PiRGBArray(self.camera, size=(self.screen_w, self.screen_h))

        for frame in self.camera.capture_continuous(rawCapture, format='rgb', use_video_port=True):
            image = frame.array
            image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # NOTE: 3d printed Swi has flipped camera to fit in the head piece
            #       flip on x (0) y (1) = -1, so opencv can get ur flipped game
            image_bw = cv2.flip(image_bw, -1)

            # human detection
            faces = self.detect_faces(image_bw)
            if (len(faces) > 0):
                text_for_humans = True
                if (face_seen_at == -1):
                    face_seen_at = frame_nb
                    face_seen_image = cv2.flip(image_bw, -1)
                #print('I see ' + str(len(faces)) + ' faces')
                #faces = self.remove_far_faces(faces)
                #if (len(faces) == 0):
                #    continue
                #x, y, w, h = self.find_closest_face(faces)
                if (draw_face_rectangles):
                    for (x, y, w, h) in faces:
                        off = int(face_offset / 2)
                        cv2.rectangle(image_bw, (x - off, y - off),
                                                (x + w + off, y + h + off),
                                                (50, 50, 50), 2)
                        #print('I see ' + 'x'.join([str(w), str(h)]) + ' of you')
            else:
                text_for_humans = False

            #print(str(face_seen_at) + ', at ' + str(frame_nb))
            if ((face_seen_at != -1) and (frame_nb > face_seen_at + wait_n_frames_swipe)):
                self.swipe(face_seen_image, choice(
                    [self.Swipe.LEFT.value, self.Swipe.RIGHT.value]
                ))
                face_seen_at = -1
                face_seen_image = None

            if (text_for_humans):
                text = self.TEXT_HUMANS
            else:
                text = self.TEXT_BINARY

            # text of poems
            if (frame_nb % text_frame_freq == 0):
                if (text_loc[1] > self.screen_h):
                    text_loc = (10, 10)
                    text_lines = list()
                if (frame_nb % max_inline_words):
                    text_loc = (text_loc[0], text_loc[1] + 20)
                    text_idle_last = frame_nb % len(self.TEXT_HUMANS)
                    text = text[text_idle_last : text_idle_last + max_inline_words]
                    text_lines.append((text_loc, ' '.join(text)))
            for line in text_lines:
                #print(line[1] + " : " + str(line[0]))
                self.text(image_bw, line[1], location=line[0])

            # flip back
            image_bw = cv2.flip(image_bw, -1)
            cv2.imshow('window', image_bw)
            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)

            if key == ord('q'):
                break

            frame_nb += 1

    def start_preview(self):
        self.camera.start_preview(window=(0, 0, self.screen_w, self.screen_h))

    def stop_preview(self):
        self.camera.stop_preview()

    def detect_faces(self, image, scale=1.1, min_neighbors=5, flags=0):
        # (image, scaleFactor, minNeighbors, flags, minSize, maxSize)
        # TODO find cvSize
        return self.face_cascade.detectMultiScale(image, scale, min_neighbors)

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
            else:
                print('Removing far away face (%i, %i, %i, %i)' % (x, y, w, h))
        return close_faces

    def swipe(self, image, direction, step_pixels=1, delay=1):
        rows, cols = image.shape
        steps = int(self.screen_w/step_pixels)
        for i in range(steps):
            M = np.float32([[1,0,i*step_pixels*direction],[0,1,0]])
            location = cv2.warpAffine(image,M,(cols,rows))
            cv2.imshow('window', location)
            cv2.waitKey(delay)

    def text(self, image, text, location=(10, 20), scale=0.5, line=2,
                   font=cv2.FONT_HERSHEY_SIMPLEX, color=(0,0,0)
    ):
        cv2.putText(image, text, location, font, scale, color, line)

    def poem(self):
        poem = list()
        nouns = "cybernetics manifesto Swi robots machines feedback humans primates technology world tray ocean thought liquid border".split()
        verbs = "transcent extend automate imitate kill reflect change branch pull sip wonder suffer steer tip".split()
        adjs = "poetic cynic symbolic radiant truthful symbiotic shocking attractive blad beautiful chubby dazzling elegant eager calm pitful".split()

        for x in range(randint(5,11)):
            w1 = ' '.join([choice(adjs) + ', ', choice(adjs)])
            w2 = ' '.join([choice(nouns), choice(verbs)])
            w3 = ' '.join([choice(nouns), choice(nouns), choice(adjs), choice(nouns)])

            for i in range(randint(2,5)):
                w = choice([w1, w2, w3])
                poem.append(w)
                poem.append(' ')

            if ((x % 3) == 0):
                poem.append(choice(nouns))
                poem.append(' the ' + choice(nouns) + choice('! ?'))
            if ((x % 7) == 0):
                w4 = ' '.join(['the ' + choice(adjs), choice(nouns), choice(verbs)])
                poem.append(w4)
                poem.append(' ')

        return ''.join(poem)

    def convert_text_binary(self, text, separator=' '):
        return separator.join(format(x, 'b') for x in bytearray(text.encode()))

if __name__ == '__main__':
    swi = Swi()
    while True:
        try:
            swi.run()
        except PiCameraValueError:
            pass
    del swi

