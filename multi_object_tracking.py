# USAGE
# python multi_object_tracking.py --video videos/soccer_01.mp4 --tracker csrt

# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import pafy
import cv2
from tkinter import *
from tkinter import ttk
import pandas as pd
import numpy as np
import datetime

# class FPS:
# 	def __init__(self):
# 		# store the start time, end time, and total number of frames
# 		# that were examined between the start and end intervals
# 		self._start = None
# 		self._end = None
# 		self._numFrames = 0

# 	def start(self):
# 		# start the timer
# 		self._start = datetime.datetime.now()
# 		return self

# 	def stop(self):
# 		# stop the timer
# 		self._end = datetime.datetime.now()

# 	def update(self):
# 		# increment the total number of frames examined during the
# 		# start and end intervals
# 		self._numFrames += 1

# 	def elapsed(self):
# 		# return the total number of seconds between the start and
# 		# end interval
# 		return (self._end - self._start).total_seconds()

# 	def fps(self):
# 		# compute the (approximate) frames per second
# 		return self._numFrames / self.elapsed()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()
# if a video path was not supplied, grab the reference to the web cam
# if not args.get("video", False):
# 	print("[INFO] starting video stream...")
# 	vs = VideoStream(src=0).start()
# 	time.sleep(1.0)
#
# # otherwise, grab a reference to the video file
# else:
# 	vs = cv2.VideoCapture(args["video"])
vs = cv2.VideoCapture('./video/multi_obj.MOV')
# fps = FPS().start()

urls = []
data_loc = './data.csv'
df = pd.read_csv(data_loc)
root = Tk()
frame_counters = []
names = []
imgs = []
option = 0
def onselect(evt):
	w = evt.widget
	global option
	index = w.curselection()[option]
	option += 1
	url = df.iloc[index, 1]
	if url not in urls:
		urls.append(url)
		name = df.iloc[index, 0]
		names.append(name)
	# for i in range(len(filenames)):
	# 	# 	if l.select_includes(i):
	# 	# 		url = df.iloc[i, 1]
	# 	# 		name = df.iloc[i, 0]
	# 	# 		urls.append(url)
	# 	# 		names.append(name)
	print('You selected item %d: "%s"' % (index, url))


def get_centroid(obejct_img,frame):
    sift = cv2.xfeatures2d.SIFT_create()
    kp2, des2 = sift.detectAndCompute(obejct_img, None)
    bf = cv2.BFMatcher()
    cam_w = frame.shape[0]  # float
    cam_h = frame.shape[1]  # float
    kp1, des1 = sift.detectAndCompute(frame, None)
    # Match descriptors.
    if des2 is not None:
        good_matches = bf.knnMatch(des1, des2, k=2)
        distance_sum = int(sum([m.distance for m, n in good_matches if m.distance < 0.75*n.distance]))
        dst_pts = np.float32([kp1[n.queryIdx].pt for m, n in good_matches]).reshape(-1, 1, 2)
        centroid = (dst_pts.sum(0) / len(dst_pts)).squeeze().astype('int')
        # frame = cv2.circle(frame, tuple(centroid), 10, (255, 255, 255), -1)
        offset = np.array([cam_w // 2, cam_h // 2]) - centroid
        curr_centroid = np.array([cam_w // 2, cam_h // 2]) - offset
        proj_centroid = curr_centroid
        x1 = proj_centroid[0] - cam_w // 8
        y1 = proj_centroid[1] - cam_h // 8
        x2 = proj_centroid[0] + cam_w // 8
        y2 = proj_centroid[1] + cam_h // 8

    return (x1,y1,x2,y2)

l = Listbox(root, selectmode=MULTIPLE, height=30, width=60)
l.grid(column=0, row=0, sticky=(N, W, E, S))
s = ttk.Scrollbar(root, orient=VERTICAL, command=l.yview)
s.grid(column=1, row=0, sticky=(N, S))
l['yscrollcommand'] = s.set
bt = Button(root, text='完成', command=root.quit)
bt.grid(row=1)
ttk.Sizegrip().grid(column=1, row=2, sticky=(S, E))
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)
root.geometry('350x500+50+50')
root.title('Select Video')
filenames = df.iloc[:, 0].tolist()
for filename in filenames:
	l.insert(END, filename)
l.bind('<<ListboxSelect>>', onselect)
root.mainloop()
root.destroy()


sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()
caps = []
# caps_cp = []
for idx in range(len(urls)):
	frame_counters.append(0)
	vPafy = pafy.new(urls[idx])
	play = vPafy.getbest()
	cap = cv2.VideoCapture(play.url)
	caps.append(cap)
	img_loc = './Photos/' + names[idx] + '/' + names[idx] + '1.jpg'
	ob_img = cv2.imread(img_loc)
	ob_img_gray = cv2.cvtColor(ob_img, cv2.COLOR_BGR2GRAY)
	# caps_cp.append(cap)
	# kp1, des1 = sift.detectAndCompute(ob_img_gray, None)
	# imgs.append((kp1, des1))

while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	# frame = frame[1] if args.get("video", False) else frame
	frame = frame[1]
	# check to see if we have reached the end of the stream
	if frame is None:
		break
	# resize the frame (so we can process it faster)
	frame = imutils.resize(frame, width=600)
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# kp1, des1 = sift.detectAndCompute(frame_gray, None)

	# grab the updated bounding box coordinates (if any) for each
	# object that is being tracked
	(success, boxes) = trackers.update(frame)
	# print(boxes)
	# loop over the bounding boxes and draw then on the frame
	for idx in range(len(boxes)):
		# (kp2, des2) = imgs[idx]
		# matches = bf.knnMatch(des1, des2, k=2)
		# good_matches = [n for m, n in matches if m.distance < 0.75*n.distance]
		# dst_pts = np.float32([kp1[n.queryIdx].pt for n in good_matches]).reshape(-1, 1, 2)
		# centroid = (dst_pts.sum(0) / len(dst_pts)).squeeze().astype('int')
		(x, y, w, h) = [int(v) for v in boxes[idx]]
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		_, video_fr = caps[idx].read()
		video_fr = cv2.resize(video_fr, (w // 2, h // 2))
		offset = w//4
		frame[y:y + h//2, x + offset:x + offset + w//2, :] = video_fr
		frame_counters[idx] += 1
		if frame_counters[idx] == caps[idx].get(cv2.CAP_PROP_FRAME_COUNT):
			frame_counters[idx] = 0
			vPafy = pafy.new(urls[idx])
			play = vPafy.getbest()
			caps[idx] = cv2.VideoCapture(play.url)
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# fps.update()

	# if the 's' key is selected, we are going to "select" a bounding
	# box to track

	if key == ord("s"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		box = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)

		# create a new object tracker for the bounding box and add it
		# to our multi-object tracker
		tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		trackers.add(tracker, frame, box)


	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break
# fps.stop()
# print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are using a webcam, release the pointer
if args.get("video", False):
	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
