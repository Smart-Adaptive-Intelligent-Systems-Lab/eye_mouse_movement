import numpy as np
import matplotlib.pyplot as plt
import cv2
import pafy
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

import sys
import numpy as np
from PIL import Image as im
from PIL import ImageEnhance as imEnh

try:
    from PIL import Image
except ImportError:
    import Image

url = "https://www.youtube.com/watch?v=iqU_BxtKP80"
# Only want to analyze first slide, so 0:00 - 0:49

# 18 vertical lines including border
# 12 horizontal lines including border 
h_offset = 60
w_offset = 64

'''
https://hackthedeveloper.com/motion-detection-opencv-python/#:~:text=Steps%20for%20Motion%20Detection%20OpenCV%20Python&text=Read%20two%20frames%20from%20the,of%20Contours%20to%20detect%20Motion.

Was looking for regions:
https://www.projectpro.io/recipes/select-and-tag-certain-regions-of-image-opencv
https://stackoverflow.com/questions/9084609/how-to-copy-a-image-region-using-opencv-in-python

"python opencv image regions"

# for video, can cut off parts of grid that the eye doesn't go to 

'''

def get_youtube_cap(url):
    play = pafy.new(url)
    # play = pafy.new(url).streams[-1] # we will take the lowest quality stream
    assert play is not None # makes sure we get an error if the video failed to load
    return cv2.VideoCapture(play.url)

# Function to extract frames
def FrameCapture(url):
    # Path to video file
    cap = cv2.VideoCapture('Condition4Gazeandmouse.mp4')
  
    # Used as counter variable
    count = 0
  
    # checks whether frames were extracted
    success = 1
  
    while success:
  
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
  
        # Saves the frames with frame-count
        cv2.imwrite("frame%d.jpg" % count, image)
  
        count += 1

        if count == 3: break
 
def draw_grid(img, rows, cols, w, h, color=(0, 255, 0), thickness=1):
    h, w, _ = img.shape
    # print(h, w) #720 1152
    # rows, cols = rows, cols
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img

def findGaze(frame1, frame2, minh, minw, cap, prev):
    # Read two frames and take the difference
    diff = cv2.absdiff(frame1, frame2)

    # Image Manipulations For Motion Detection OpenCV Python
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Finding Contours In OpenCV
    contours, _ = cv2.findContours( dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    ox = 0 
    oy = 0

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        ox, oy = applyGridOffset(x, y)
        (b, g, r) = frame1[np.int16(y/2), np.int16(x/2)]

        #region Capture Gaze
        # Red rectangle is the cursor 
        if w < 43 or h < 43:
        # if w < 30 or h < 30:
            minw.append((h, w))
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # cv2.imwrite("movement/cursor(x" + str(x) + "y" + str(y) + ")__(h" + str(h) +"w" +str(w) +")" + ".jpg", frame1)
            color = (0, 0, 255)
            # print('Red: (h{}, w{})'.format(h,w))
            cv2.putText(frame1, "{} at ({},{}) with size (h{} w{}) and b{}, g{}, r{}".format('Movement', ox, oy, h, w, b, g, r), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # These coordinates are the yellow circle gaze 
        else:
            minh.append((h, w)) 
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # cv2.imwrite("movement/gaze(x" + str(x) + "y" + str(y) + ")__(h" + str(h) +"w" +str(w) +")" + ".jpg", frame1)

            color = (255, 0, 0)
            cv2.putText(frame1, "{} at ({},{}) with size (h{} w{}) and b{}, g{}, r{}".format('Movement', ox, oy, h, w, b, g, r), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # cap.get(cv2.CAP_PROP_POS_MSEC)
            print('"{},{}", {}'.format(oy, ox, int(cap.get(cv2.CAP_PROP_POS_MSEC))))

        i += 1
        #endregion

def applyGridOffset(x, y):
    ox = 0
    oy = 0

    if x / w_offset > 0: ox = x / w_offset
    if y / h_offset > 0: oy = y / h_offset
    return int(round(ox)), int(round(oy))

if __name__ == "__main__":
    # image = cv2.imread("frame0.jpg")

    batch_size = 16

    # cap = get_youtube_cap(url)
    videoName = 'Condition_4_Gaze_and_mouse.mp4'
    # videoName = 'Condition_3_Gaze_no_mouse.mp4'
    # videoName = 'Condition_2_No_gaze_mouse.mp4'

    cap = cv2.VideoCapture(videoName)
    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    size = min([width, height])

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    # out = cv2.VideoWriter("out49.avi", fourcc, 20, (size, size))
    # out = cv2.VideoWriter('out49.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (size, size))
    out = cv2.VideoWriter('output.avi',fourcc, 20.0,(int(cap.get(3)),int(cap.get(4))))

    timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
    calc_timestamps = [0.0]
    minw = []
    minh = []

    frame_exists, frame = cap.read()
    prev = (2, 8)

    print('point, ms')

    #region Works for whole video
    while(cap.isOpened()):
        frame_exists, frame = cap.read()
        frame_exists, frame2 = cap.read()
        if frame_exists:
            framev2 = draw_grid(frame, 12, 18, width, height)
            frame2v2 = draw_grid(frame2, 12, 18, width, height)
            findGaze(frame, frame2, minh, minw, cap, prev)

            # Write edited frame to new video
            out.write(frame)

            # Display the resulting frame
            cv2.imshow('Frame',frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            calc_timestamps.append(calc_timestamps[-1] + 1000/fps)

            if int(cap.get(cv2.CAP_PROP_POS_MSEC)) >= 49000: 
                print("First 49000 milliseconds is the first 49 seconds i.e. first slide of video")
                break
            #endregion
        else:
            break

    cap.release()
