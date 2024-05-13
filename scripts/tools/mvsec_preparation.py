import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

import h5py
import hdf5plugin
import numpy as np

from mvsec_helper import *

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/root/data/MVSEC', type=str)
    parser.add_argument('--scenario', default='indoor_flying', type=str)
    args = parser.parse_args()
    
    # Get calibartion and retification informations
    calib_dir = args.data_dir + '/' + args.scenario + '/' + args.scenario + '_calib'
    calib, rect_map = get_calibration_info(args.scenario, calib_dir)
    focal_length_x_baseline = calib['cam1']['projection_matrix'][0][3]
    
    for exp in EXPERIMENTS[args.scenario]:
        data_path = args.data_dir + '/' + args.scenario + '/' + args.scenario + str(exp) + '_data.hdf5'
        gt_path = args.data_dir + '/' + args.scenario + '/' + args.scenario + str(exp) + '_gt.hdf5'
        save_dir = args.data_dir + '/' + args.scenario + '/' + args.scenario + str(exp)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        assert Path(data_path).exists()
        assert Path(gt_path).exists()
        assert Path(save_dir).is_dir()
        
        # Load the files
        data = h5py.File(data_path, 'r')
        gt = h5py.File(gt_path, 'r')
        
        events = {}
        images = {}
        event_timestamps = {}
        image_timestamps = {}
        for loc in LOCATION:
            # Get the events
            events[loc] = np.array(data['davis'][loc]['events'])        # EVENTS: X Y TIME POLARITY
            event_timestamps[loc] = events[loc][:,2].tolist()
            # Get the images
            if args.scenario == 'outdoor_day':
                # A hardware failure caused the grayscale images on the right DAVIS 
                # grayscale images for this scene to be corrupted. 
                # However, VI-Sensor grayscale images are available
                images[loc] = np.array(data['visensor'][loc]['image_raw'])
                image_timestamps[loc] = np.array(data['visensor'][loc]['image_raw_ts']).tolist()
            else:
                images[loc] = np.array(data['davis'][loc]['image_raw'])
                image_timestamps[loc] = np.array(data['davis'][loc]['image_raw_ts']).tolist()
        # Get the depth gt
        depth_gt = np.array(gt['davis']['left']['depth_image_rect'])
        sync_timestamps = np.array(gt['davis']['left']['depth_image_rect_ts']).tolist()
        assert depth_gt.shape[0] == len(sync_timestamps)
        
        # Rectifying events and images
        # Iterate for left & right cameras
        for cam, loc in zip(['cam0', 'cam1'], LOCATION):
            # 'cam0': left camera, 'cam1': right camera
            if args.scenario == 'outdoor_day' and cam == 'cam0':
                cam = 'cam2'
            elif args.scenario == 'outdoor_day' and cam == 'cam1':
                cam = 'cam3'
            rectified_to_distorted_x, rectified_to_distorted_y = get_rectification_map(calib[cam])
            image_size = calib[cam]['resolution']
            
            # Iterate for depth timestamps
            event_idx_t = 0
            image_idx_t = 0
            for synchronization_index, end_timestamp in tqdm(enumerate(sync_timestamps), total=len(sync_timestamps), desc=(args.scenario + str(exp) + f'/{loc}')):
                file_name = str(synchronization_index).zfill(6)    # e.g., 000002, 000004, ...
                start_timestamp = end_timestamp - TIME_BETWEEN_EXAMPLES
                
                # Get synchronized events
                for t in range(event_idx_t, len(event_timestamps[loc]), 1):
                    # Find event start index
                    event_timestamp = events[loc][t,2]
                    if event_timestamp >= start_timestamp:
                        event_start_index = t
                        event_idx_t = t
                        break
                for t in range(event_idx_t, len(event_timestamps[loc]), 1):
                    # Find event end index
                    event_timestamp = events[loc][t,2]
                    if event_timestamp > end_timestamp:
                        event_end_index = t
                        event_idx_t = t
                        break
                synchronized_events = events[loc][event_start_index:event_end_index]
                
                # Events rectification
                rectified_synchronized_events = np.array(rectify_events(synchronized_events, rect_map[loc]['x'], rect_map[loc]['y'], image_size))
                
                # Save events as .npy
                events_save_dir = save_dir + f'/events/{loc}'
                if not os.path.exists(events_save_dir):
                    os.makedirs(events_save_dir)
                np.save(events_save_dir + f'/{file_name}', rectified_synchronized_events)
                
                # Get synchronized raw image
                time_diff = np.finfo(np.float64).max
                for t in range(image_idx_t, len(image_timestamps[loc]), 1):
                    if np.abs(end_timestamp - image_timestamps[loc][t]) <= time_diff:
                        time_diff = np.abs(end_timestamp - image_timestamps[loc][t])
                    else:
                        # Find synchronized image index
                        image_sync_index = t - 1
                        image_idx_t = t
                        break
                synchronized_image = images[loc][image_sync_index]
                
                # Image rectification
                rectified_synchronized_image = cv2.remap(synchronized_image, rectified_to_distorted_x, rectified_to_distorted_y, cv2.INTER_LINEAR)
                
                # Save image as .png
                image_save_dir = save_dir + f'/images/{loc}'
                if not os.path.exists(image_save_dir):
                    os.makedirs(image_save_dir)
                cv2.imwrite(image_save_dir + f'/{file_name}.png', rectified_synchronized_image)
                
                # Convert depth map to disparity map
                depth_gt[synchronization_index][np.isnan(depth_gt[synchronization_index])] = 0.
                disp_gt = depth2disparity(depth_gt[synchronization_index], focal_length_x_baseline)
                
                # Save disparity map as .png
                save_gt_dir = save_dir + '/disparity_gt'
                if not os.path.exists(save_gt_dir):
                    os.makedirs(save_gt_dir)
                cv2.imwrite(save_gt_dir + f'/{file_name}.png', disp_gt)
                
            assert len(os.listdir(events_save_dir)) == len(os.listdir(image_save_dir)) == len(os.listdir(save_gt_dir))
        print(f'{args.scenario + str(exp)} finish')
    print('Data saving as individual files completed')