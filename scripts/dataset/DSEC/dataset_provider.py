from pathlib import Path

import torch
import torch.utils.data

from .sequence import Sequence
from .constant import DATA_SPLIT


class DatasetProvider:
    def __init__(self, data_dir: str, modality: str, representation: str='voxel', delta_t_ms: int=50, num_bins: int=5, 
                 height=480, width=640, 
                 crop_height=432, crop_width=576, pad_height=480, pad_width=648):
        self.data_dir = data_dir
        
        self.modality = modality
        self.representation = representation
        self.delta_t_ms = delta_t_ms
        self.num_bins = num_bins
        
        self.height = height
        self.width = width
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.pad_height = pad_height
        self.pad_width = pad_width
        
    def get_train_dataset(self):
        sequence_list = DATA_SPLIT['train']
        train_sequences = list()
        for child in sequence_list:
            child = self.data_dir + '/' + child
            assert Path(child).is_dir()
            train_sequences.append(Sequence(seq_dir=child, mode='train', 
                                            modality=self.modality, representation=self.representation, 
                                            delta_t_ms=self.delta_t_ms, num_bins=self.num_bins, 
                                            height=self.height, width=self.width, 
                                            edit_height=self.crop_height, edit_width=self.crop_width))
            
        train_dataset = torch.utils.data.ConcatDataset(train_sequences)
        return train_dataset
    
    def get_valid_dataset(self):
        sequence_list = DATA_SPLIT['validation']
        valid_sequences = list()
        for child in sequence_list:
            child = self.data_dir + '/' + child
            assert Path(child).is_dir()
            valid_sequences.append(Sequence(seq_dir=child, mode='validation', 
                                            modality=self.modality, representation=self.representation, 
                                            delta_t_ms=self.delta_t_ms, num_bins=self.num_bins, 
                                            height=self.height, width=self.width, 
                                            edit_height=self.pad_height, edit_width=self.pad_width))
            
        valid_dataset = torch.utils.data.ConcatDataset(valid_sequences)
        return valid_dataset
    
    def get_test_dataset(self):
        sequence_list = DATA_SPLIT['test']
        test_sequences = list()
        for child in sequence_list:
            child = self.data_dir + '/' + child
            assert Path(child).is_dir()
            test_sequences.append(Sequence(seq_dir=child, mode='test', 
                                           modality=self.modality, representation=self.representation, 
                                           delta_t_ms=self.delta_t_ms, num_bins=self.num_bins, 
                                           height=self.height, width=self.width, 
                                           edit_height=self.pad_height, edit_width=self.pad_width))
            
        test_dataset = torch.utils.data.ConcatDataset(test_sequences)
        return test_dataset