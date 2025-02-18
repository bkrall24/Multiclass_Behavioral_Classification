import numpy as np
import cv2
import pandas as pd
import os


def frame_crop(x, y, buffer = 200, shape = [1080, 1920]):
    """ determines appropriate crop coordinates based on center, buffer, and shape of frame

    Args:
        x (numerical): x coordinate of center to crop
        y (numericakl): y coordinate of center to crop
        buffer (int, optional): buffer on each side of center (i.e. 2 * buffer = length). Defaults to 200.
        shape (list, optional): shape of frame. Defaults to [1080, 1920].

    Returns:
        center_coordinates: coordinates to use for crop [min y, max y, min x, max x]
        center: rounded x, y
    """
    if (x + buffer) >  shape[1]:
        max_x = shape[1]
    else:
        max_x = x+buffer

    if (x - buffer) < 0:
        min_x = 0
    else:
        min_x = x-buffer

    if (y + buffer) >  shape[0]:
        max_y = shape[0]
    else:
        max_y = y+buffer

    if (y - buffer) < 0:
        min_y = 0
    else:
        min_y = y - buffer

    # print(f"Min y: {min_y}, Max y: {max_y}, Min x: {min_x}, Max x: {max_x}")
    cc = [round(min_y, 1), round(max_y, 1), round(min_x,1), round(max_x,1)]
    center = [round(x, 1),round(y, 1)]
    return [round(c) for c in cc], [round(c) for c in center]

def padcrop_image(frame, buffer, cc, center):
    """Crops image based on center and distance from center (buffer) 
       adds a black border to account for end of frame

    Args:
        frame (matrix): video frame (cv2.read() output)
        buffer (int): size to crop from center coords
        cc (list): min and max y and x for cropping
        center (list): center x and y coordinates

    Returns:
        frame: cropped frame with any necessary black padding
    """
    padded_image = np.zeros((buffer*2, buffer*2, 3), dtype=np.uint8)
     
    # Paste the cropped region onto the black image
    padded_image[
        max(0, buffer - (center[1] - cc[0])):min(buffer*2, buffer + (cc[1] - center[1])),
        max(0, buffer - (center[0] - cc[2])):min(buffer*2, buffer + (cc[3] - center[0])),:] = frame[cc[0]:cc[1], cc[2]:cc[3],:]

    return padded_image

def get_params_from_vid(vid_path, animals = ['m1','m2','m3','m4','m5','m6'], dp = 0.5, minDist = 400):
    # NEED TO UPDATE TO ALLOW YOU TO HANDLE MORE CASES THAN 6 mice or 1 mouse

    """ Returns video details, finds the center of circular cylinders used to contain mice in videos to allow for cropping

    Args:
        vid_path (str): filepath to video
        animals (list, optional): list of animals - useful for vids with 6 mice. Defaults to ['m1','m2','m3','m4','m5','m6'].
        dp (float, optional): parameter for matching circles. Defaults to 0.5.
        minDist (int, optional): parameter for matching circles. Defaults to 400.

    Returns:
        _type_: _description_
    """
    src = cv2.VideoCapture(str(vid_path))
    frames = src.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = src.get(cv2.CAP_PROP_FPS)

    ret = False
    count = 0

    while not ret:
        ret, frame = src.read() 
        count = count +1
        if count > frames:
            return None
    src.release()
     
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist)
    # print(circles)

    if len(animals) > 1:
        found = match_circles(circles[0])
        dat = {}
        for f, a in zip(found[0], animals):
            dat[a] = f
    else:
        dat = {}
        if circles is not None:
            x_dif = np.argmin(abs(circles[0][:,0] - (frame.shape[1]/2)))
            dat[animals] = circles[0][x_dif,:]
        else:
            dat[animals] = np.array([frame.shape[1]/2, frame.shape[0]/2, 350])
        # plot_circles(vid_path, np.array( [[dat['1']]]), radius = 400)
    return dat, frames, fps, frame.shape

def match_circles(circles, x_exp = np.array([350, 980, 1560]), y_exp = np.array([285, 800]), distance = 100):
    """ Matches circles detected using HoughCircles to expected location and arrangement of arenas in mouse videos

    Args:
        circles (cv2.HoughCircles): output of cv2.HoughCirlces
        x_exp (numpy array, optional): list of expected x coordinates for circles. Defaults to np.array([350, 980, 1560]).
        y_exp (numpy array, optional): list of expected y coordinates for circles. Defaults to np.array([285, 800]).
        distance (int, optional): Threshold for distance between circle and expected location. Defaults to 100.

    Returns:
        _type_: _description_
    """
    found = np.full([1,6,3], np.nan)
    x_displace = np.zeros((6))
    y_displace = np.zeros((6))
    for c in circles:
        # print(c)
        x_dis = x_exp - c[0]
        x_match = np.where(np.abs(x_dis) <= distance)[0]

        y_dis = y_exp - c[1]
        y_match = np.where(np.abs(y_exp - c[1]) <= distance)[0]

        
        if len(x_match) and len(y_match):
            id = (3* y_match)+ x_match 
            found[0,id,:] = c
            x_displace[id]= x_dis[x_match[0]]
            y_displace[id] = y_dis[y_match[0]]

    x_displace[x_displace == 0] = np.nan
    y_displace[y_displace == 0] = np.nan

    for v in np.argwhere(np.isnan(found)):
        # print(v[1])
        if v[2] == 0: # x
            start = np.mod(v[1], 3)
            dis = np.nanmean(x_displace[start::3])
            if np.isnan(dis):
                dis = np.nanmean(x_displace)
            found[v[0], v[1], v[2]] = x_exp[start] - dis
        if v[2] == 1: # y
            if v[1] < 3:
                dis = np.nanmean(y_displace[:3])
                row = 0
            else:
                dis = np.nanmean(y_displace[3:])
                row = 1
            found[v[0], v[1], v[2]] = y_exp[row] - dis

        if v[2] == 2: # r
            found[v[0], v[1], v[2]] = np.nanmean(found[0,:,2])

        
    return found

def plot_circles(vid_path, circles, radius = None):

    src = cv2.VideoCapture(str(vid_path))
    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT) 

    ret = False
    count = 0

    while not ret:
        ret, frame = src.read() 
        count = count +1
        if count > video_length:
            return None
          
    src.release()
	# convert the (x, y) coordinates and radius of the circles to integers
    v = np.round(circles).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
    # for v in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
    if radius is None:
        radius = v[2]
    v = v.astype(int)
    cv2.circle(frame, (v[0], v[1]), radius, (0, 255, 0), 4)
    cv2.rectangle(frame, (v[0] - 5, v[1] - 5), (v[0] + 5, v[1] + 5), (0, 128, 255), -1)
    # show the output image
    cv2.imshow("output", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def determine_crop_specs(video_path, num_animals = None):

    if num_animals == 6:
        crop_specs =  {'dp':  0.5, 'minDist': 400, 'animals': ['m1','m2','m3','m4','m5','m6']}
    else:
        src = cv2.VideoCapture(str(video_path))
        ret, frame = src.read()
        frame_size = frame.shape

        if frame_size == (1080, 1920, 3) or num_animals == 6:
            crop_specs =  {'dp':  0.5, 'minDist': 400, 'animals': ['m1','m2','m3','m4','m5','m6']}
        else:
            crop_specs = {'dp': 0.5, 'minDist': 400, 'animals': [1]}
    
    return crop_specs