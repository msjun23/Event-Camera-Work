from pathlib import Path
import weakref
import csv
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .eventslicer import EventSlicer
from ..representations import *
from .. import transforms


class Sequence(Dataset):
    # NOTE: This is just an EXAMPLE class for convenience. Adapt it to your case.
    # In this example, we use the voxel grid representation.
    #
    # In this structure, seq/images/left/ev_aligned/ is not contained in original dataset
    # This folder have to be made by align_ev_im.py
    #
    # This class assumes the following structure in a sequence directory:
    #
    # seq_name (e.g. zurich_city_11_a)
    # ├── calibration
    # │   ├── cam_to_cam.yaml
    # │   └── cam_to_lidar.yaml
    # ├── disparity
    # │   ├── event
    # │   │   ├── 000000.png
    # │   │   └── ...
    # │   ├── image
    # │   │   ├── 000000.png
    # │   │   └── ...
    # │   └── timestamps.txt
    # ├── events
    # │   ├── left
    # │   │   ├── events.h5
    # │   │   └── rectify_map.h5
    # │   └── right
    # │       ├── events.h5
    # │       └── rectify_map.h5
    # └── images
    #     ├── left
    #     │   ├── rectified
    #     │   │   ├── 000000.png
    #     │   │   ├── We do not use these images
    #     │   │   └── Prepare aligned image data before running the train script
    #     │   ├── ev_aligned
    #     │   │   ├── 000000.png
    #     │   │   └── ...
    #     │   └── exposure_timestamps.txt
    #     ├── right
    #     │   ├── rectified
    #     │   │   ├── 000000.png
    #     │   │   ├── We do not use these images
    #     │   │   └── Prepare aligned image data before running the train script
    #     │   ├── ev_aligned
    #     │   │   ├── 000000.png
    #     │   │   └── ...
    #     │   └── exposure_timestamps.txt
    #     └── timestamps.txt
    
    def __init__(self, seq_dir: str, mode: str='train', 
                 modality: str='event', representation: str='voxel', 
                 delta_t_ms: int=50, num_bins: int=15, 
                 height=480, width=640, 
                 edit_height=480, edit_width=640):
        assert Path(seq_dir).is_dir()
        assert mode in ['train', 'validation', 'test']
        assert modality in ['event', 'image', 'EI']
        assert delta_t_ms <= 100, 'adapt this code, if duration is higher than 100 ms'
        assert num_bins >= 1
        
        ################
        # Base setting #
        ################
        # NOTE: Adapt this code according to the present mode (e.g. train, val or test).
        self.mode = mode
        
        # Set data transforms
        if self.mode in ['train']:
            self.transforms = transforms.Compose([
                transforms.RandomCrop(crop_height=edit_height, crop_width=edit_width), 
                transforms.RandomVerticalFlip(), 
            ])
        elif self.mode in ['validation', 'test']:
            self.transforms = transforms.Compose([
                transforms.Padding(pad_height=edit_height, pad_width=edit_width)
            ])
        else:
            raise NotImplementedError
        
        # Save sequence info
        self.seq_dir = seq_dir
        self.seq_name = seq_dir.split('/')[-1]
        
        # Save output dimensions
        self.channel = num_bins
        self.height = height
        self.width = width
        
        # Set output modality
        self.modality = modality
        
        # Save delta timestamp in us
        self.delta_t_us = delta_t_ms * 1000
        
        #############
        # Load data #
        #############
        # Load gt
        self.load_gt_path()
        
        # Set event representation
        self.representation = representation
        if self.representation == 'voxel':
            self.representation_method = VoxelGrid(self.channel, self.height, self.width, normalize=True)
        elif self.representation == 'on_off':
            self.representation_method = OnOffFrame(self.channel, self.height, self.width)
        elif self.representation == 'raw':
            self.representation_method = RawEvent(self.channel, self.height, self.width)

        # Set for stereo setup
        self.locations = ['left', 'right']
        
        # Load event(image) data
        if self.modality == 'event':
            # Load only event and disparity label
            self.load_event_data()
            self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)
        elif self.modality == 'image':
            # Load only image and disparity label
            self.load_image_path()
        elif self.modality == 'EI':
            # Load event, image and disparity label
            self.load_event_data()
            self.load_image_path()
            self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)
        else:
            print('========== Check your modality setting at config file! ==========')
            raise NotImplementedError
            
    def load_event_data(self):
        # Load h5f files -> event data
        self.h5f = dict()
        self.rectify_ev_maps = dict()
        self.event_slicers = dict()
        ev_dir = Path(self.seq_dir) / 'events'
        for location in self.locations:
            ev_dir_location = ev_dir / location
            ev_data_file = ev_dir_location / 'events.h5'
            ev_rect_file = ev_dir_location / 'rectify_map.h5'
            
            h5f_location = h5py.File(str(ev_data_file), 'r')
            self.h5f[location] = h5f_location
            self.event_slicers[location] = EventSlicer(h5f_location)
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]
                
    def load_image_path(self):
        # Load images path
        self.rectified_image_pathstrings = dict()
        img_dir = Path(self.seq_dir) / 'images'
        for location in self.locations:
            img_dir_location = img_dir / location / 'ev_aligned'
            assert img_dir_location.is_dir()
            rectified_image_pathstrings = list()
            for img in img_dir_location.iterdir():
                assert str(img.name).endswith('.png')
                if self.mode in ['train', 'validation'] and int(str(img.stem)) % 2 == 0:
                    rectified_image_pathstrings.append(str(img))
                elif self.mode in ['test'] and int(str(img.stem)) % 20 == 0:
                    rectified_image_pathstrings.append(str(img))
            rectified_image_pathstrings.sort()
            assert int(Path(rectified_image_pathstrings[0]).stem) == 0
            # Remove first component
            rectified_image_pathstrings.pop(0)
            self.rectified_image_pathstrings[location] = rectified_image_pathstrings
            
    def load_gt_path(self):
        self.timestamps = np.array([])
        if self.mode in ['train', 'validation']:
            # load disparity timestamps
            disp_dir = Path(self.seq_dir) / 'disparity'
            assert disp_dir.is_dir()
            self.timestamps = np.loadtxt(disp_dir / 'timestamps.txt', dtype='int64')
            
            # load event disparity paths
            ev_disp_dir = disp_dir / 'event'
            assert ev_disp_dir.is_dir()
            event_disp_gt_pathstrings = list()
            for entry in ev_disp_dir.iterdir():
                assert str(entry.name).endswith('.png')
                event_disp_gt_pathstrings.append(str(entry))
            event_disp_gt_pathstrings.sort()
            self.event_disp_gt_pathstrings = event_disp_gt_pathstrings
            assert len(self.event_disp_gt_pathstrings) == self.timestamps.size
            
            ################################################
            # By aligning image to event camera resolution #
            # image gt does not required                   #
            ################################################
            
            # Remove first disparity path and corresponding timestamp.
            # This is necessary because we do not have events before the first disparity map.
            assert int(Path(self.event_disp_gt_pathstrings[0]).stem) == 0
            self.event_disp_gt_pathstrings.pop(0)
            self.timestamps = self.timestamps[1:]
        elif self.mode == 'test':
            # load test events timestamps
            test_csv_dir = Path(self.seq_dir) / Path(Path(self.seq_dir).stem+'.csv')
            csv_f  = open(test_csv_dir, 'r')
            next(csv.reader(csv_f))
            self.test_file_index = np.array([])
            for l in csv.reader(csv_f):
                self.timestamps = np.append(self.timestamps, np.array([int(l[0])], dtype='int64'), axis=0)
                self.test_file_index = np.append(self.test_file_index, np.array([int(l[1])], dtype='int32'), axis=0)
                
    def events_representation_method(self, x, y, p, t):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        
        return self.representation_method.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))
        
    def getHeightAndWidth(self):
        return self.height, self.width
    
    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32')/256
    
    @staticmethod
    def get_image_data(filepath: Path):
        assert filepath.is_file()
        image_16bit = cv2.imread(str(filepath), cv2.IMREAD_COLOR)   # BGR
        image_16bit = cv2.cvtColor(image_16bit, cv2.COLOR_BGR2RGB)  # RGB
        return image_16bit.astype('float32')/256
    
    @staticmethod
    def close_callback(h5f_dict):
        for k, h5f in h5f_dict.items():
            h5f.close()
            
    def __len__(self):
        if self.mode in ['train', 'validation']:
            return len(self.event_disp_gt_pathstrings)
        else:
            return self.timestamps.size
        
    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps[location]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]
    
    def __getitem__(self, index):
        ts_end = self.timestamps[index]
        ts_start = ts_end - self.delta_t_us
        output = {}
        
        if self.mode in ['train', 'validation']:
            # These datasets have disparity info(gt)
            event_disp_gt_path = Path(self.event_disp_gt_pathstrings[index])
            event_disparity_gt = torch.from_numpy(self.get_disparity_map(event_disp_gt_path))
            file_index = int(event_disp_gt_path.stem)
            output['disparity_gt'] = event_disparity_gt
            output['file_index'] = file_index
            output['seq_name'] = self.seq_name
        elif self.mode == 'test':
            # Test set do not have gt
            file_index = int(self.test_file_index[index])
            output['file_index'] = file_index
            output['seq_name'] = self.seq_name
            
        for location in self.locations:
            # Load event data
            if self.modality == 'event' or self.modality == 'EI':
                event_data = self.event_slicers[location].get_events(ts_start, ts_end)
                
                p = event_data['p']
                t = event_data['t']
                x = event_data['x']
                y = event_data['y']
                
                xy_rect = self.rectify_events(x, y, location)
                x_rect = xy_rect[:, 0]
                y_rect = xy_rect[:, 1]
                
                event_representation = self.events_representation_method(x_rect, y_rect, p, t)
                if 'event' not in output:
                    output['event'] = dict()
                output['event'][location] = event_representation
            
            # Load image data
            if self.modality == 'image' or self.modality == 'EI':
                if 'image' not in output:
                    output['image'] = dict()
                rectified_image_path = Path(self.rectified_image_pathstrings[location][index])
                rectified_image = torch.from_numpy(np.copy(self.get_image_data(rectified_image_path)))
                rectified_image = np.transpose(rectified_image, (2,0,1))
                
                output['image'][location] = rectified_image
        
        # Apply transforms
        output = self.transforms(output)
        
        return output