import numpy as np
import pandas as pd
import random
from video_preprocessing import get_params_from_vid, padcrop_image, frame_crop
from src.prediction import predict_from_table, load_experiment_table
from ast import literal_eval
import cv2
import os
import pickle
import datetime

## currently only works with binary labels - need to adapt for more

def rolling_analysis(logits, window):
    """ Calculate a rolling average and count for the predictions. Used to determine clips 
        to sample for manual validation"""
    avg = np.array([np.sum(logits[i:i+window])/window for i in range(len(logits-window))])
    counts = np.array([np.sum(logits[i:i+window] < 0)/window for i in range(len(logits-window))])

    return avg, counts
    
def find_starts_by_confidence(logits, window):
    """ Determines start index to sample clips for validation"""
    avg, counts = rolling_analysis(logits, window)
    off = np.mean(logits[logits > 0])
    on = np.mean(logits[logits < 0])

    starts = {}
    a = np.where((avg >= (0.80 * off)) & (counts == 0))
    b = np.where((avg <= (0.80 * on)) & (counts == 1))
    starts['high_off'] = a[0]
    starts['high_on'] = b[0]

    c = np.where((avg >= (0.4 * off)) & (avg < (0.80 * off)) &  (counts == 0))
    d = np.where((avg <= (0.4 * on)) & (avg > (0.80 * on)) & (counts == 1))
    starts['mid_off'] = c[0]
    starts['mid_on'] = d[0]

    e = np.where((avg >= 0) & (avg < (0.4 * off)) & (counts == 0))
    f = np.where((avg <= 0) & (avg > (0.4 * on)) & (counts == 1))
    starts['low_off']= e[0]
    starts['low_on'] = f[0]

    return starts

def find_starts_bins(logits, window, bins = 10):
    """ Pulls starts for clips """
    avg, counts = rolling_analysis(logits, window)
    num, edges = np.histogram(avg, bins = bins)

    starts = {}
    for low,high in zip(edges[:-1], edges[1:]):
        a = np.where((avg >= low) & (avg < high) & ((counts % 1) == 0))
        starts[low] = a[0]
    
    return starts, num

def grab_clip_with_annotation(cap, start, window, logits, prediction_frames, center, buffer, frame_delay, add_text = True):
    """ Function that uses prediction, starts and length to return clips """
    clip_logits = logits[start:start+window]
    frame_start = int(start * prediction_frames) + frame_delay

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    clip = []
    for l in clip_logits:
        for _ in range(prediction_frames):
            ret, frame = cap.read()

            cc, center = frame_crop(center[0], center[1], buffer = buffer, shape = frame.shape[:2])
            pc = padcrop_image(frame, buffer, cc, center)

            if (l < 0) and add_text:
                cv2.putText(pc, "scratching", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
            clip.append(pc)

    return clip

def show_clip_grab_input(clip, labels, clip_name, delay = 10):
    """ Shows a clip via cv2 and grabs user input"""
    replay = True
    ind_choice = None
    while replay:
        for frame in clip:
            cv2.imshow(f'{clip_name}', frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                replay = False
            elif key == 32:
                while (ind_choice not in [str(i) for i in np.arange(0, len(labels))]):
                    ind_choice = input([f'{ind}: {item}' for ind, item in enumerate(labels)])
                    if ind_choice == "":
                        ind_choice = None
                        break
                replay = False
    
    return ind_choice

def validate_prediction(video_path, animal, logits, prediction_frames, window, delay_start = 0, num_clips = 10, bins = 10, labels = ['Not scratch', 'scratch', 'mixed'], show_label = False):
    """ Parent function that takes video, animal id and prediction to choose and present clips for 
        manual validation of model scoring. 
    """

    # Determine necessary parameters to get appropriate video clips
    circles, vid_frames, fps, _ = get_params_from_vid(video_path)
    frame_delay = delay_start * 60 * fps
    center = circles[animal]
    cap = cv2.VideoCapture(video_path)

    # Determine sets of frames of high, medium, and low confidence labeling
    # starts = find_starts_by_confidence(logits, window)
    starts, counts = find_starts_bins(logits, window, bins = bins)

    # Initialize a dictionary to store labeled clips
    user_labled_clips ={k: [] for k in labels}

   
    presumed_wrong = 0
    all_possible = 0

    all_wrong = 0
    all_examined = 0 
    # iterate through the different 'confidence' levels for each label
    for ind,(k,v) in enumerate(starts.items()):
        # print(f'Evaluating Bin: {ind} with {counts[ind]} videos')
        print(f'Evaluating Bin: {ind} with {len(v)} videos')
        # choose a subset of clips
        if len(list(v)) > num_clips:
            sample_options = list(v) # this needs to be shuffled then popped
            random.shuffle(sample_options)
            start_sample = []
            # count = 0

            while (len(start_sample) < num_clips) and (len(sample_options) != 0):
                # samp = random.sample(list(v),1)
                samp = sample_options.pop(0)
                if all([abs(x - samp) > window for x in start_sample]):
                    start_sample.append(samp)

            start_sample = random.sample(list(v), num_clips)
        else:
            start_sample = list(v)
        start_sample.sort()

        wrong = 0
        total = 0
        # iterate through the clips
        for s in start_sample:

            # Determine the clips 'confidence' i.e. proportion of frames labeled 1 
            confidence = sum(logits[s:s+window] < 0) / window ## THIS LINE WILL NEED TO CHANGE IF DOING MULTI-LABEL

            # Get the clip, show it to the user, prompt them to choose 
            clip = grab_clip_with_annotation(cap, s, window, logits, prediction_frames, center, 300, frame_delay, add_text = show_label)
            user_label = show_clip_grab_input(clip, labels, f'{labels[int(np.round(confidence))]}')

            # if the user actually labeled the clip, determine if it was labeled wrong
            # add it to dictionary of clips
            if user_label:
                total = total + 1
                if int(user_label) != int(np.round(confidence)):
                    wrong = wrong + 1

                user_labled_clips[labels[int(user_label)]].append(clip)

            cv2.destroyAllWindows()
            cv2.waitKey(1)

        print(f'Wrong: {wrong}, Total {total}')
        if (counts[ind] != 0) & (total != 0):
        # if len(v):
            presumed_wrong = presumed_wrong + ((wrong/total) * counts[ind])
            # presumed_wrong = presumed_wrong + ((wrong/total) * len(v))
            all_wrong = all_wrong + wrong
            all_examined = all_examined + total
            all_possible = all_possible + counts[ind]
            # all_possible = all_possible + len(v)
            print(f'Wrong Count: {presumed_wrong}')
            print(f'All Count: {all_possible}')

    
    cap.release()
    percent_presumed_wrong = (presumed_wrong/all_possible) * 100
    percent_wrong = (all_wrong/all_examined) * 100

    return user_labled_clips, percent_presumed_wrong, percent_wrong

def write_clip(clip, save_path, fps, size):
    
    """ Writes clip to folder, for augmenting training set"""

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frame in clip:
        out.write(frame)
    out.release()
    print(f'Saved: {save_path}')

def calculate_validation_csv_metrics(scored):
    """ Uses output of predict_from_table on a set validation table to generate scores
    """
    scores = {}
    num_bins = max([int(x.split('Bin ')[-1]) for x in scored.columns if ': Bin' in x])
    for x in range(num_bins):
        scored['dif '+str(x+1)] = scored['bin '+str(x+1)] - scored['Duration: Bin '+str(x+1)]

    
    difs = scored.filter(like = 'dif')

    
    scored['abs sum'] = difs.abs().sum(axis = 1)
    scores['Full Video: Abs difference mean'] = scored['abs sum'].mean()
    scores['Full Video: Abs difference median'] = scored['abs sum'].median()
    scores['Full Video: Abs difference top'] = scored['abs sum'].max()
    scores['Worst Video'] = scored['File Name'].iloc[scored['abs sum'].argmax()]+ '_' + str(scored['Subject #'].iloc[scored['abs sum'].argmax()])
    scores['Bin: Max Difference'] = difs.max(axis = None)
    scores['Bin: Min Difference'] = difs.min(axis = None)
    scores['Bin: Mean Difference'] = difs.mean(axis = None)
    scores['Bin: Median Difference'] = difs.median(axis = None)
    scores['Bin: FP count'] = (difs < 0).sum().sum() 
    scores['Bin: FN count'] = (difs > 0).sum().sum() 
    scores['Bin: FP percent'] =  (scores['Bin: FP count'] /difs.count().sum()) * 100
    scores['Bin: FN percent'] =  (scores['Bin: FN count'] /difs.count().sum()) * 100

    dif_array = np.array(difs)
    scores['Bin: FP average'] =np.mean(dif_array[dif_array < 0])
    scores['Bin: FP median'] = np.median(dif_array[dif_array < 0])

    scores['Bin: FN average'] =np.mean(dif_array[dif_array > 0])
    scores['Bin: FN median'] = np.median(dif_array[dif_array > 0])

    if "Elapsed Minutes" in scored.columns:
        scores['mean run time'] = scored['Elapsed Minutes'].mean()
    else:
        scores['mean run time'] = None

    return scored, scores

def calculate_validation_ann_csv(csv_filename, validation_folder, save_folder, num_frames, save = False):
    """ Uses videos not in dataset to test model """
    df, meta = load_experiment_table(csv_filename)

    prediction_folder = os.path.join(save_folder, 'predictions')
    ann_scores = []
    for ind, row in df.iterrows():

        ann_file = os.path.join(validation_folder, 'Validated_' + row['File Name'] + '_' + str(row['Subject #']) + '.csv')
        pred_file = os.path.join(prediction_folder, row['File Name'] + '_' + str(row['Subject #']) + '.pkl')
        with open(pred_file, 'rb') as f:
            prediction = pickle.load(f)

        via_output = pd.read_csv(ann_file, skiprows = 9)
        tc = via_output['temporal_coordinates']
        start_time = row['Scoring Start (min)'] * 60
        _, _, fps, _ = get_params_from_vid(os.path.join(row['Folder Path'], row['File Name']))
        tc2 = np.array([(np.array(literal_eval(a)) ) for a in tc])- start_time
        tp = np.array([np.round(a * fps ).astype(int) for a in tc2])
        clip_length = row['Scoring End (min)'] - row['Scoring Start (min)']
        frames = int(fps * 60 * clip_length)
        scratch = np.zeros(frames)
        for j in tp:
            scratch[j[0]:j[1]] = [1] * (j[1] - j[0])
        
        max_frame = len(prediction) * num_frames
        ann_bin = np.mean(np.reshape(scratch[:max_frame], (-1, num_frames)), 1)

        pred = np.squeeze(prediction) < 0
        ann = ann_bin > 0.3
        difference = pred[pred != ann]
        fp = difference.sum()
        fn = len(difference) - fp
        ann_scores.append({'File Name': row['File Name'], 'Subject #': row['Subject #'], 
                           'fp count': fp, 'fp percent': (fp/len(pred))*100, 'fn count': fn, 
                           'fn percent': (fn/len(pred))*100})
    
    
    scored_df = pd.DataFrame(ann_scores)
    if save:
        output_file = os.path.join(save_folder, 'scored_' + os.path.basename(csv_filename))
        scored_df.write_csv(output_file)

    return scored_df

def validate_model(model, model_specs, data_specs, validation_folder, save_folder, save = True):
    """ Full validation of a model on standard validation dataset"""
    if save_folder is None:
        parent_folder = os.path.dirname(model_specs['checkpoint_dir'])
        model = os.path.basename(parent_folder)
        os.makedirs(os.path.join(parent_folder, model + '_validations'), exist_ok = True)
        day = datetime.now().strftime("%m%d%y")
        if os.path.exists(os.path.join(parent_folder, model + '_validations', model + '_validation_'+ day)):
            day_with_time = datetime.now().strftime("%m%d%y_%H%M")
            save_folder = os.path.join(parent_folder, model + '_validations', model + '_validation_'+ day_with_time)
        else:
            save_folder = os.path.join(parent_folder, model + '_validations', model + '_validation_'+ day)
    
        os.makedirs(save_folder, exist_ok = True)

    csvs = [x for x in os.listdir(validation_folder) if 'ann' not in x.lower() and '.DS' not in x and 'Validated' not in x]
    all_scores = []
    for c in csvs:
        output = predict_from_table(os.path.join(validation_folder, c), model, model_specs, data_specs, save_folder, timed = True)
        scored = pd.read_csv(output)
        scored, scores = calculate_validation_csv_metrics(scored)
        scored.to_csv(output)
        all_scores.append(scores)

    ann = [x for x in os.listdir(validation_folder) if 'ann' in x.lower() and '.DS' not in x and 'Validated' not in x]
    all_ann_scores = []
    for a in ann:
        ann_score  = calculate_validation_ann_csv(os.path.join(validation_folder, a), validation_folder, save_folder, data_specs['num_frames'])
        all_ann_scores.append(ann_score)
   
   
    final_ann_scores = pd.concat(all_ann_scores)
    final_manual_scores = pd.DataFrame(all_scores)

    if save:
        final_manual_scores.to_csv(os.path.join(save_folder, 'manual_scores.csv'))
        final_ann_scores.to_csv(os.path.join(save_folder, 'ann_scores.csv'))

    return final_ann_scores, final_manual_scores
        

    




   
    


     