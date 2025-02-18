import os
import pickle
import random
import cv2 

import tensorflow as tf, tf_keras
import numpy as np

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

from dataset_generation import format_frames
from video_preprocessing import get_params_from_vid, padcrop_image, frame_crop

def load_movinet_model( resolution = 172, num_classes = 1, input_shape = [8,30,172,172,3], checkpoint_path = "trained_model/cp.ckpt"):

    model_id = 'a0'
    use_positional_encoding = model_id in {'a3', 'a4', 'a5'}
    resolution = resolution

    # Create backbone and model.
    backbone = movinet.Movinet(
        model_id=model_id,
        causal=True,
        conv_type='2plus1d',
        se_type='2plus3d',
        activation='hard_swish',
        gating_activation='hard_sigmoid',
        use_positional_encoding=use_positional_encoding,
        use_external_states=True,
    )

    model = movinet_model.MovinetClassifier(
        backbone,
        num_classes= num_classes,
        output_states=True)

    # Create your example input here.
    # Refer to the paper for recommended input shapes.
    inputs = tf.ones(input_shape)

    # [Optional] Build the model and load a pretrained checkpoint.
    model.build(inputs.shape)

    # Load weights from the checkpoint to the rebuilt model
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    # model.load_weights(checkpoint_path)
    
    # checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    # checkpoint = tf.train.Checkpoint(model=model)
    # status = checkpoint.restore(checkpoint_path)

    return model


class ClipDictGenerator:
    def __init__(self, video_path, time_dict, bin = 15, n_frames = 5, buffer = 200, output_size = (172,172)):
        """ Returns a set of frames with their associated label. 

        Args:
            df_path: path to dataframe
            indices: indices of df to use for 
            n_frames: Number of frames. 
            training: Boolean to determine if training dataset is being created.
        """

        self.video_path = video_path
        self.time_dict = time_dict
        self.buffer = buffer
        self.output_size = output_size
        self.bin = bin
        self.n_frames = n_frames
        self.circles, self.total_frames, self.fps, _ = get_params_from_vid(self.video_path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        

    def __call__(self):

        start = int(np.min([v[0] for v in self.time_dict.values()]) * self.fps  * 60)
        stop = int(np.max([v[1] for v in self.time_dict.values()]) * self.fps * 60)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        rand_frames = random.sample(range(0, self.bin), self.n_frames)
        
        count = -1
        for _ in range(int((stop-start)/self.bin)):
            result = {k: [] for k in self.time_dict.keys()}
            for bin_frame in range(self.bin):
                ret, frame = self.cap.read()
                count = count + 1
                if bin_frame in rand_frames:
                    for k,v in self.time_dict.items():
                        if (count < (v[1] * self.fps * 60)) & (count >= (v[0]* self.fps * 60)):
                            
                            center = self.circles[k]

                            cc, center = frame_crop(center[0], center[1], buffer = self.buffer, shape = frame.shape[:2])
                            pc = padcrop_image(frame, self.buffer, cc, center)

                            # crop_frame = padcrop_image(frame, self.buffer, center[:2]) ## these likely need to change to match FrameGenerator 
                            result[k].append(format_frames(pc, self.output_size))
            
            result = {k: np.array(v) for k,v in result.items() if len(v)}
            yield result
        
        self.cap.release()


    def close_vid(self):
        self.cap.release()

def predict_video_streamwise(vid_file, time_dict, cg_params, model_params, save_dir = None):

    cg = ClipDictGenerator(video_path = vid_file, time_dict = time_dict, **cg_params)

    model = load_movinet_model(**model_params)
    initial_state = model.init_states([1, cg_params['n_frames'], cg_params['output_size'][0], cg_params['output_size'][1], 3])
    
    if save_dir is None:
        save_dir = os.path.dirname(vid_file)

    fns = {}
    for k in time_dict.keys():
        fn = os.path.join(save_dir, k+'_predict.pkl')
        fns[k] = open(fn, 'wb')

    # dump metadata first
    # video_file
    # timing (i,e, start, stop)
    # expected shape - implement len in clipdictgenerator
    # probably eventually want to output 

    state = initial_state.copy()
    for ind, frames in enumerate(cg()):
        inputs = state
        for k,v in frames.items():
            inputs['image'] = v[tf.newaxis, ...]
            logits, _ = model(inputs)
            # all_logits[k].append(logits)

            pickle.dump(logits, fns[k])

    for k in time_dict.keys():
        fns[k].close()
    
    # if model_params['n_classes'] < 3:
    #     preds = tf.round(tf.nn.sigmoid(all_logits))
    # else:
    #     preds = tf.nn.softmax(all_logits)


    return True




class VidGenerator:
    def __init__(self, video_path, n_frames = 1, frame_step = 1, buffer = 300, output_size = (172,172), 
                 animal_choice = ['m1'], dp = 0.5, minDist = 400, start = 0, stop = None):
        """ Returns a set of frames with their associated label. 

        Args:
            df_path: path to dataframe
            indices: indices of df to use for 
            n_frames: Number of frames. 
            training: Boolean to determine if training dataset is being created.
        """
        self.video_path = video_path
        self.batch_size = n_frames * frame_step
        self.buffer = buffer
        self.output_size = output_size
        circles, self.vid_frames, self.fps, _ = get_params_from_vid(self.video_path, dp = dp, minDist = minDist)
        self.circles = circles[animal_choice]
        self.step = frame_step
        self.start = start
        if stop is None:
            self.stop = self.vid_frames
        else:
            self.stop = stop

    def __call__(self):

        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
        print('opened cap')
        
        for _ in range(int((self.stop-self.start)/self.batch_size)):
            result = []
            for i in range(self.batch_size):
                ret, frame = cap.read()
                if i % self.step == 0:

                    center = self.circles

                    cc, center = frame_crop(center[0], center[1], buffer = self.buffer, shape = frame.shape[:2])
                    pc = padcrop_image(frame, self.buffer, cc, center)
                    result.append(format_frames(pc, self.output_size))
                    
                    # center = self.circles # test the output of get_params_from_vide
                    # frame = padcrop_image(frame, self.buffer, center[:2])
                    # result.append(format_frames(frame, self.output_size))
            
            result = np.array(result)
            yield result[tf.newaxis,...]
            
        
        cap.release()