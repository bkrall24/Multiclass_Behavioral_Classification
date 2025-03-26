# predict a video using a segmentation model
import string
import json
import time
import random
import csv
from video_preprocessing import *
from model_loading import *
import tensorflow as tf, tf_keras

# Create a dataset generator designed for our videos
# Just for a single animal - might want to edit to be able to cut all the animals from a six animal video
class PredictClipGenerator:
    """ Generator for pulling frames from videos to predict"""
    def __init__(self, video_path, data_specs, crop_specs, animal_choice = 'm1', start = 0, stop = None):
        """ Returns a set of frames with their associated label. 

        Args:
            video_path: path to video
            data_specs: dict containing necessary data parameters for model - number of frames, frame step (i.e. 2 would skip every other frame), buffer, output size
            video_specs: dict containing necessary video parameter to isolate mouse - dp, minDist, animals for determining circle for centering
            animal_choice: to choose correct value from circles for cropping. 
            start, stop = time in mintues for choosing start and stop of prediction

        """
        self.video_path = video_path
        self.predict_size = data_specs['num_frames'] * data_specs['frame_step']
        self.buffer = data_specs['buffer']
        self.output_size = data_specs['output_size']
        circles, self.vid_frames, self.fps, _ = get_params_from_vid(self.video_path, animals = crop_specs['animals'],
                                                                    dp = crop_specs['dp'], minDist = crop_specs['minDist'])
        self.circles = circles[animal_choice]
        self.step = data_specs['frame_step']
        self.start = start * 60 * self.fps
        if stop is None:
            self.stop = self.vid_frames
        else:
            self.stop = stop * 60 * self.fps

    def __call__(self):

        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
        print('opened cap')
        
        for j in range(int((self.stop-self.start)/self.predict_size)):
            result = []
            for i in range(self.predict_size):
                ret, frame = cap.read()
                if i % self.step == 0:
                    
                    center = self.circles

                    cc, center = frame_crop(center[0], center[1], buffer = self.buffer, shape = frame.shape[:2])
                    pc = padcrop_image(frame, self.buffer, cc, center)
                    pc = tf.image.convert_image_dtype(pc, tf.float32)
                    pc = tf.image.resize_with_pad(pc, *self.output_size)

                    result.append(pc)
                    
                    # center = self.circles # test the output of get_params_from_vide
                    # frame = padcrop_image(frame, self.buffer, center[:2])
                    # result.append(format_frames(frame, self.output_size))
            
            result = np.array(result)
            yield result[tf.newaxis,...]
            
        
        cap.release()

# takes a loaded model and affiliated data_specs, a video and its crop_specs, and details about the animal
# and generates predictions
def predict_video(model, data_specs, video_path, crop_specs, animal, start, stop):
    """ Predicts from the video using the model"""
    data = PredictClipGenerator(video_path, data_specs, crop_specs, animal, start, stop)
    fps = data.fps
    predict_data = tf.data.Dataset.from_generator(data, output_signature = (tf.TensorSpec(shape = (None, None, None, None, 3), dtype = tf.float32)))
    predictions = model.predict(predict_data, batch_size = 256)

    return predictions, fps

# takes a boolean derived from predictions and generates stop/stop times for each
# instance of the behavior. Requires the number of frames used for predictions (predict_frames)
# and fps of original video
def boolean_to_annotations(y, predict_frames, fps):
    """Generates start and stop time in seconds for a boolean array
        Assumes that y is a boolean (1, 0) where 1 is the behavior of interest.
        Predict frames augments each prediction to the correct number of frames it represents
        Uses fps to determine timestamps for changes. 
        Can be used for multi-behavior labeling but each behavior is inputted as boolean separately

    Args:
        y (numpy.ndarray): boolean array indicating frames with scratching
        fps (int, optional): frame rate

    Returns:
        list: list of starts, stops for each bout of scratching in boolean array
    """
    y = np.array([[x] * predict_frames for x in y]).flatten()

    y = np.insert(y, 0,0)
    starts = np.where(np.diff(y) == 1)[0] 
    stops = np.where(np.diff(y)== -1)[0] 

    seconds = np.linspace(0, len(y)/fps, len(y))
    st = seconds[starts]
    sto = seconds[stops]
    tc2 = [[s1, s2] for s1,s2 in zip(st,sto)]
    return tc2

# takes annotations (output of boolean to annotations) and determines duration
# and number of bouts of behavior in bins
def bin_annotations(anns, start = 0, length = 30, bin_size = 5):
    """ Takes annotations (i.e. start stop of behavior) and generates list
        of behavior bouts and duration in a given bin (defaults to 5 min bin)
    """
    n_bins = int(length/bin_size)
    all_bouts = []
    all_durations = []
    for s in range(n_bins):
        start_sec = (s * bin_size * 60) + (start * 60)
        end_sec = start_sec + (bin_size * 60)

        
        duration = 0
        bouts = 0
        for a in anns:
            b = (a >= start_sec) & (a <= end_sec)
            if b[0] & b[1]:
                duration = duration + (a[1] - a[0])
                bouts = bouts + 1
            if b[1] & ~b[0]:
                duration = duration +(a[1] - start_sec)
                bouts = bouts + 1
            if b[0] & ~b[1]:
                duration = duration +(end_sec - a[0])
                bouts = bouts + 1
        
        all_bouts.append(bouts)
        all_durations.append(duration)

    return all_durations, all_bouts


# following three functions build a json file to load annotations into VIA
# ann_dict: keys are either behavior names (scratch, shake) with corresponding annotations (from boolean_to_annotation)
#   or keys are animal names with corresponding single behaviors (i.e m1, m2 but all containing scratching annotations)
def generate_random_key(exceptions =[]):
    """ random key to create a JSON for VIA annotation software"""
    characters = string.ascii_letters + string.digits
    unique = False

    while not unique:
        gen_string = ''.join(random.choice(characters) for _ in range(8))
        unique = gen_string not in exceptions

    return gen_string

def format_via_annotation(ann_dict):
    """ Formats data into VIA JSON to review in VIA annotation software"""
    exceptions = []
    metadata = {}
    for animal in ann_dict:
        for a in ann_dict[animal]:
            next_key = generate_random_key(exceptions)
            exceptions.append(next_key)
            full_key = "1_"+next_key
            metadata[full_key] =  {'vid': '1',
                'flg': 0,
                'z': a.tolist(),
                'xy': [],
                'av': {'1': animal}}
    
    return metadata

def build_json_dict(fn, ann_dict, save_file):

    project = {'pid': '__VIA_PROJECT_ID__',
        'rev': '__VIA_PROJECT_REV_ID__',
        'rev_timestamp': '__VIA_PROJECT_REV_TIMESTAMP__',
        'pname': 'Unnamed VIA Project',
        'creator': 'VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via)',
        'created': int(time.time() *1000),
        'vid_list': ['1']}
    
    config = {'file': {'loc_prefix': {'1': '', '2': '', '3': '', '4': ''}},
        'ui': {'file_content_align': 'center',
        'file_metadata_editor_visible': True,
        'spatial_metadata_editor_visible': True,
        'temporal_segment_metadata_editor_visible': True,
        'spatial_region_label_attribute_id': '',
        'gtimeline_visible_row_count': '4'}}
    

    attribute = {'1': {'aname': 'TEMPORAL-SEGMENTS',
        'anchor_id': 'FILE1_Z2_XY0',
        'type': 4,
        'desc': 'Temporal segment attribute added by default',
        'options': {'default': 'Default'},
        'default_option_id': ''}}
    
    file = {'1': {'fid': '1',
        'fname': fn,
        'type': 4,
        'loc': 1,
        'src': ''}}
    
    metadata = format_via_annotation(ann_dict)

    view = {'1': {'fid_list': ['1']}}

    json_dict = {'project':project, 'config':config, 'attribute':attribute, 'file':file, 'metadata':metadata, 'view': view}


    with open(save_file, "w") as outfile: 
        json.dump(json_dict, outfile)

    return json_dict

# load the template I designed for storing reference for a given experiment
def load_experiment_table(table_file, header_size = 6):
    """ Loads data from template for a given project"""
    ext = os.path.splitext(table_file)[1]
    if ext == '.csv':
        header = pd.read_csv(table_file, nrows = header_size)
        tbl = pd.read_csv(table_file, header = header_size+1)

        header = header.transpose().reset_index()
        header = header.loc[~header['index'].str.contains('Unnamed'),:].dropna(axis = 1)
    elif 'xls' in ext:
        header = pd.read_excel(table_file, nrows = header_size)
        tbl = pd.read_excel(table_file, header= header_size + 1)
        header = header.transpose()
    else:
        print('Could not read file type '+ ext)
        return None
    
    
    header.columns = header.iloc[0]
    metadata = dict(header.iloc[-1])

    tbl.loc[tbl['Folder Path'].isna(),'Folder Path'] = metadata['Folder Path']

    if int(metadata['Number of animals per video']) == 6:
        key = {1:'m3', 2: 'm6', 3: 'm2', 4: 'm5', 5: 'm1', 6: 'm4'}
        tbl['Animal ID'] = tbl['Arena Location'].map(key)
    else:
        tbl['Animal ID'] = tbl['Arena Location']
    
    return tbl, metadata

def predict_from_table(table_file, model, model_specs, data_specs, output_folder, header_size = 6, bin_size = 5, timed = False):
    """ Generates prediction from a csv table"""
    tbl, metadata = load_experiment_table(table_file=table_file, header_size=header_size)

    columns = list(tbl.columns)
    clip_length_max = (tbl['Scoring End (min)'] - tbl['Scoring Start (min)']).max().item()
    for c in range(int(clip_length_max/bin_size)):
        columns.append('Duration: Bin '+str(c+1))
        columns.append('Bouts: Bin '+str(c+1))
    columns.append('Duration: Total')
    columns.append('Bouts: Total')
    if timed:
        columns.append('Elapsed Minutes')
    output_file = os.path.join(output_folder, 'scored_' + os.path.basename(table_file))
    with open(output_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames = columns)
        writer.writeheader()

    # all_predictions = {}

    with open(output_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        for ind,row in tbl.iterrows():
            if timed:
                start_time = time.time()

            video_path = os.path.join(row['Folder Path'], row['File Name'])
            crop_specs = determine_crop_specs(video_path, metadata['Number of animals per video'])
            predictions, fps = predict_video(model, data_specs, video_path, crop_specs, row['Animal ID'], row['Scoring Start (min)'], row['Scoring End (min)'])
            # all_predictions[ind] = predictions
            clip_length = row['Scoring End (min)'] - row['Scoring Start (min)']
            os.makedirs(os.path.join(output_folder, 'predictions'), exist_ok=True)
            pred_file = os.path.join(output_folder,'predictions', row['File Name']+ '_'+ str(row['Subject #']) + '.pkl')

            with open(pred_file, 'wb') as pf:
                pickle.dump(predictions, pf)
            
            if timed:
                end_time = time.time()
            
            out_row = dict(row)
            if model_specs['num_classes'] == 1:
                bool_predict = np.array(predictions < 0).astype(int)
                ann = boolean_to_annotations(bool_predict, data_specs['num_frames'], fps)
                duration, bouts = bin_annotations(np.array(ann), start = 0, length = clip_length, bin_size = bin_size) # since we start the predictions at the scoring start - don't delay.
                for ind, (d,b) in enumerate(zip(duration, bouts)):
                    out_row['Duration: Bin '+str(ind+1)] = d
                    out_row['Bouts: Bin '+str(ind+1)] = b
                out_row['Duration: Total'] = sum(duration)
                out_row['Bouts: Total'] = sum(bouts)
            if timed:
                out_row['Elapsed Minutes'] = (end_time-start_time) / 60
            # all_scores.append(out_row)
            writer.writerow(out_row)


    return output_file


    
    

    
## base model predicts in clips of x frames and returns the prediction average for x frames. Streaming models
# generate predictions per frame using the previous frame output as input. Previously this made it so that 


# state = initial_state.copy()
# for ind, frames in enumerate(cg()):
#     inputs = state
#     for k,v in frames.items():
#         inputs['image'] = v[tf.newaxis, ...]
#         logits, _ = model(inputs)
        






