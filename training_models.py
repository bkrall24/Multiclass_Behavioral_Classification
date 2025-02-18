# training functions
import random
import glob
import pandas as pd
import cv2
from model_loading import *

def format_frames(frame, output_size, rotate = 0, reflect = False):
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
    if reflect:
        frame = tf.image.flip_left_right(frame)
    if rotate != 0:
        frame = tf.image.rot90(frame, k = rotate)

    return frame

def frames_from_video_file(video_path, n_frames, rotate = 0, reflect = False, output_size = (172,172), frame_step = 1):
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
  result.append(format_frames(frame, output_size, rotate = rotate, reflect = reflect))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size, rotate = rotate, reflect = reflect)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)

  return result

def generate_clip_table(folder):
    """
    Retrieve the files that belong to each class.

    Args:
        files: List of files in the dataset.

    Return:
        Dictionary of class names (key) and files (values).
    """
    behaviors = list(glob.glob(folder +'/*'))
    beh_list = []
    file_list = []
    for beh in behaviors:
        files = list(glob.glob(beh + '/*.mp4'))
        name = os.path.basename(beh)
        file_list.extend(files)
        beh_list.extend([name] * len(files))
    
    return pd.DataFrame({'behavior': beh_list , 'filepaths': file_list})

def choose_behavior_subset(folder, classes, splits, shuffle = True, mult_max = 7):
    """
    Create a directory for a subset of the dataset and split them into various parts, such as
    training, validation, and test.

    Args:
        folder: folder containing data.
        classes: dict defining classes. Folder contains subfolders of behaviors that can be grouped (i.e. body and facial grooming)
        splits: Dictionary specifying the training, validation, test, etc. (key) division of data
                (value is number of files per split).
        shuffle: Boolean if videos should be shuffle
        mult_max: maximum possible ways to augment dataset if not enough clips. 7 are iterations of reflect and 90, 180, and 270 degree rotations

    Return:
        pandas dataframe containing 
    """

    clip_table = generate_clip_table(folder)
    class_key = {item:k for k,v in classes.items() for item in v}
    clip_table['class'] = clip_table['behavior'].map(class_key)
    clip_table = clip_table.dropna()

    files_needed = sum(splits.values()) 
    
    all_clips = []
    for class_name in clip_table['class'].unique():
        df = clip_table.loc[clip_table['class'] == class_name]
        mult = np.ceil(files_needed / len(df)).astype(int)

        if mult > mult_max:
            print(f'Too few files in class: {class_name} even with augmentation')
            return None
        
        df2 = pd.concat([df for _ in range(mult)])
        df2['augment_key'] = np.array([[i] * len(df) for i in range(mult)]).flatten()
        if shuffle:
           df2 = df2.sample(frac = 1)
        df2 = df2.reset_index(drop = True)

        df2 = df2.iloc[:files_needed, :]
        sp = []
        [sp.extend([k] * v) for k,v in splits.items()]
        df2['Split'] = sp

        all_clips.append(df2)

    final_clips = pd.concat(all_clips)
    final_clips = final_clips.sample(frac = 1)
    
    return final_clips

def interpret_augment_key(all_clips, rotate_key = {0: 0, 1:0, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3}, 
                          reflect_key = {ind: (ind%2) != 0 for ind in range(7)}):

    all_clips['reflect'] = all_clips['augment_key'].map(reflect_key)
    all_clips['rotate'] = all_clips['augment_key'].map(rotate_key)

    return all_clips

class FrameGenerator:
  def __init__(self, file_table, n_frames, frame_step, class_ids, shuffle = True):
    """ Returns a set of frames with their associated label.

      Args:
        file_dict: ouptput of choose behavior subset
        n_frames: Number of frames.
        frame_step: sampling (every 1 frame = 1, every other = 2)

    """
    self.file_table = file_table
    self.frame_step = frame_step
    self.n_frames = n_frames
    self.class_names = list(file_table['class'].unique())
    # self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))
    self.class_ids_for_name = class_ids

  def __call__(self):
    for _, row  in self.file_table.iterrows():
      video_frames = frames_from_video_file(row['filepaths'], self.n_frames, frame_step = self.frame_step, rotate = row['rotate'], reflect = row['reflect'])
      label = self.class_ids_for_name[row['class']] # Encode labels
    #   print(f"rotate: {row['rotate']}, reflect: {row['reflect']}")
      yield video_frames, label

def generate_datasets(all_clips, num_frames, frame_step, class_labels = None):

    output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))
    
    if class_labels is None:
        class_labels = dict((name, idx) for idx, name in enumerate(list(all_clips['class'].unique())))
    datasets = {}
    for split in all_clips['Split'].unique():
       datasets[split] = tf.data.Dataset.from_generator(FrameGenerator(all_clips.loc[all_clips['Split'] == split,:], num_frames, frame_step, class_labels), 
                                                        output_signature = output_signature)
       
    
    return datasets, class_labels

def train_seg_model(model, model_specs, data_specs, data_folder, classes, splits, checkpoint_path, save_weights_only = True, save_best_only = True,
                    monitor = 'val_accuracy', mode = 'max', epochs = 10, verbose = 1):
    # inputs 
    #   model, model_specs - from load_seg_model or create_seg_model
    #   data_specs - can be used from loaded model, or change to fit needs of training - output size == (res, res) from model_specs['resolution']
    #   data_folder - folder with behavior folders with clips
    #   classes - definition of how to lump behaviors 
    #   splits - number of videos to have in each train/test/val 
    #   checkpoint path - where to save the new model weights

    # generate a table defining test, train, val sets of the classes of interest
    all_clips = choose_behavior_subset(data_folder, classes, splits)
    all_clips = interpret_augment_key(all_clips)

    # ensure that your class labels match previous trained models if using the same classes
    if 'class_labels' in model_specs.keys():
        data_classes = set(all_clips['class'].unique())
        model_classes = set(model_specs['class_labels'].keys())

        class_labels = {}
        for c in data_classes.intersection(model_classes):
            class_labels[c] = model_specs['class_labels'][c]
        ind = len(class_labels)
        for c in data_classes.difference(model_classes):
           class_labels[c] = ind
           ind = ind + 1
    else:
       class_labels = None
        
    # generate a dataset 
    datasets_dict, class_ids = generate_datasets(all_clips, data_specs['num_frames'], data_specs['frame_step'], class_labels=class_labels)

    # generate callbacks to save model weights as it trains
    checkpoint_dir = os.path.dirname(checkpoint_path)
    new_checkpoint_path = os.path.join(checkpoint_dir, 'cp_'+date.today().strftime("%m%d%y"))

    cp_callback = tf_keras.callbacks.ModelCheckpoint(filepath=new_checkpoint_path,
                                                    save_weights_only=save_weights_only,
                                                    monitor = monitor,
                                                    mode = mode,
                                                    verbose=1,
                                                    save_best_only = save_best_only)

    batch_size = model_specs['batch_size']
    if 'train' in datasets_dict.keys():
        train_ds = datasets_dict['train']
        train_ds = train_ds.batch(batch_size)
    else:
        print('need to define a train split in splits')
        return None
    
    if 'val' in datasets_dict.keys():
        val_ds = datasets_dict['val']
        val_ds =val_ds.batch(batch_size)
    else:
        val_ds = None

    # train the model
    history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=epochs,
                    validation_freq=1,
                    verbose=verbose,
                    callbacks = [cp_callback])
    
    if save_best_only:
        if mode == 'max':
            ind = history.history[monitor].index(max(history.history[monitor]))
        else:
           ind = history.history[monitor].index(min(history.history[monitor]))
    else:
        ind = -1

    
    # test the model
    if 'test' in datasets_dict.keys():
        test_ds = datasets_dict['test']
        test_ds = test_ds.batch(batch_size)
        test_loss, test_accuracy = model.evaluate(test_ds)
    else:
        test_loss = None
        test_accuracy = None


    model_specs['checkpoint_dir'] = checkpoint_dir
    model_specs['checkpoint_path'] = new_checkpoint_path
    model_specs['num_frames'] = data_specs['num_frames']

    if len(classes.keys()) == 2:
        model_specs['num_classes'] = 1
    else:
        model_specs['num_classes'] = len(classes.keys())
    model_specs['class_labels'] = class_ids

    accuracy = {'train': history.history['accuracy'][ind], 'test': test_accuracy, 'val': history.history['val_accuracy'][ind]}
    loss = {'train': history.history['loss'][ind], 'test': test_loss, 'val': history.history['val_loss'][ind]}

    return model, model_specs, data_specs, accuracy, loss, all_clips


# save_segmentation_model(save_path, model_specs, data_specs, accuracy, loss, test_train_split)


# notes
# training saves a checkpoint. In order to use those checkpoints for effective
# model use, we need to know the model details



# options for checkpoints to save whole model or just weights
# currently saving weights only and building backbone each time
# we can also use following code (theoretically) to load model directly

# def load_model_directly(model):
#    model.compile(loss=..., optimizer=...,
#               metrics=['accuracy'])

#     EPOCHS = 10
#     checkpoint_filepath = '/tmp/ckpt/checkpoint.model.keras'
#     model_checkpoint_callback = tf_keras.callbacks.ModelCheckpoint(
#         filepath=checkpoint_filepath,
#         monitor='val_accuracy',
#         mode='max',
#         save_best_only=True)

#     # Model is saved at the end of every epoch, if it's the best seen so far.
#     model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

#     # The model (that are considered the best) can be loaded as -
#     tf_keras.models.load_model(checkpoint_filepath)