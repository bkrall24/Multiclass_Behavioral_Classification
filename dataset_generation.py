import os
import glob
import random
import cv2
import numpy as np
import imageio
from tensorflow_docs.vis import embed
import tensorflow as tf, tf_keras


def split_class_lists(files_for_class, count):
  """
    Returns the list of files belonging to a subset of data as well as the remainder of
    files that need to be downloaded.

    Args:
      files_for_class: Files belonging to a particular class of data.
      count: Number of files to download.

    Return:
      split_files: Dictionary of the files in subset of data
      remainder: Dictionary of the remainder of files that need to be downloaded.
  """
  split_files = {}
  remainder = {}
  for cls in files_for_class.keys():
    split_files[cls] = files_for_class[cls][:count]
    remainder[cls] = files_for_class[cls][count:]
  return split_files, remainder

def choose_behavior_subset(folder, classes, splits):
  """
    Create a directory for a subset of the dataset and split them into various parts, such as
    training, validation, and test.

    Args:
      folder: folder containing data.
      class_choices: dict defining classes. Folder contains subfolders of behaviors that can be grouped (i.e. body and facial grooming)
      splits: Dictionary specifying the training, validation, test, etc. (key) division of data
              (value is number of files per split).

    Return:
      dir: Posix path of the resulting directories containing the splits of data.
  """

  files_for_class = get_files_per_class(folder)

  new_files_for_class = {}
  for k,v in classes.items():
      files =  [item for subclass in v for item in files_for_class[subclass]]
      random.shuffle(files)
      new_files_for_class[k] = files


  dirs = {}
  for split_name, split_count in splits.items():
    # print(split_name, ":")
    split_files, new_files_for_class = split_class_lists(new_files_for_class, split_count)
    dirs[split_name] = split_files

  return dirs

def get_files_per_class(folder):
  """
    Retrieve the files that belong to each class.

    Args:
      files: List of files in the dataset.

    Return:
      Dictionary of class names (key) and files (values).
  """
  behaviors = list(glob.glob(folder +'/*'))
  files_for_class = {}
  for beh in behaviors:
      files_for_class[os.path.basename(beh)] = list(glob.glob(beh + '/*.mp4'))
  
  return files_for_class

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (172,172), frame_step = 15):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)

  return result

class FrameGenerator:
  def __init__(self, file_dict, n_frames, frame_step, training = False):
    """ Returns a set of frames with their associated label.

      Args:
        path: Video file paths.
        n_frames: Number of frames.
        training: Boolean to determine if training dataset is being created.
    """
    self.file_dict = file_dict
    self.frame_step = frame_step
    self.n_frames = n_frames
    self.training = training
    self.class_names = list(file_dict.keys())
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def get_files_and_class_names(self):
    video_paths = []
    classes = []
    for k, v in self.file_dict.items():
        video_paths.extend(v)
        classes.extend([k] * len(v))
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()

    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      video_frames = frames_from_video_file(path, self.n_frames, frame_step = self.frame_step)
      label = self.class_ids_for_name[name] # Encode labels
      yield video_frames, label