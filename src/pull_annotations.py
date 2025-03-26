## dealing with annotations or via or whatever
import cv2
import os
from video_preprocessing import *

def read_temporal_annotations(annotation_file, fps = 30, skiprows = 1):
    """Read annotations as seconds for given animals in a file

    Args:
        annotation_file (str): path to annotation file
        skiprows (int, optional): number of rows to skip in annotation file. Defaults to 2.

    Returns:
        dict: list of start, stops for each animal 
    """
    via_output = pd.read_csv(annotation_file, skiprows = skiprows)
    via_output['behavior'] = via_output['metadata'].apply(lambda x: x.split('"TEMPORAL-SEGMENTS":"')[-1][:-2].title())
    via_output['frame_start'] = round(via_output['temporal_segment_start']*fps).astype(int)
    via_output['frame_end'] = round(via_output['temporal_segment_end']*fps).astype(int)
    
    return via_output.loc[:,['behavior', 'frame_start', 'frame_end', 'temporal_segment_start', 'temporal_segment_end']].sort_values(by = 'frame_start').reset_index(drop = True)

def generate_behavior_clips(vid, ann_table, folder, crop_center= None, buffer = None, frame_shape = [1080, 1920, 3]):
    """ Generate clips based on annotations

    Args:
        vid (str): filepath to video
        ann_table (pandas df): table containing annotation start, end - output of read temporal annotations
        folder (str): folderpath to save clips to
        crop_center (list, optional): center of animal if cropping. Defaults to None.
        buffer (int, optional): buffer around center to crop. Defaults to None.
        fps (int, optional): fps of video. Defaults to None.
        frame_shape (list, optional): Shape of video. Defaults to [1080, 1920, 3].
    """
    writers = {k:None for k in ann_table['behavior'].unique()}
    cap = cv2.VideoCapture(vid)

    fps = cap.get(cv2.CAP_PROP_FPS)
    start = ann_table.loc[0, 'frame_start']

    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start-1))
    frame_count = start
    reading = True

    if buffer is not None:
        out_shape = (buffer *2, buffer *2, 3)
    else:
        out_shape = frame_shape

    while reading:
        current_table = ann_table.loc[(frame_count < ann_table['frame_end']) & (frame_count >= ann_table['frame_start']),:]
        ret, frame = cap.read()
        if not ret:
            print(f'Could not read frame {frame_count}')
            break 

        for _, row in current_table.iterrows():
        
            if (frame_count == row['frame_start']) or (writers[row['behavior']] is None):
                write_name = os.path.join(folder, row['behavior'], os.path.basename(vid)+ '_'+ str(row['frame_start'])+ '_'+str(row['frame_end']) +'.mp4')
                writer = cv2.VideoWriter(write_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, out_shape[:2])
                writers[row['behavior']] = writer
            else:
                writer = writers[row['behavior']]
            

            if crop_center is not None:
                cc, center = frame_crop(crop_center[0], crop_center[1], buffer = buffer, shape = out_shape[:2])
                pc = padcrop_image(frame, buffer, cc, center)
                writer.write(pc)
            else:
                writer.write(frame)
            
            if (frame_count + 1) > row['frame_end']:
                writer.release()
                writers[row['behavior']] = None
    

        frame_count = frame_count + 1
        if frame_count > ann_table.loc[ann_table.shape[0]-1, 'frame_end']:
            reading = False
    
    cap.release()