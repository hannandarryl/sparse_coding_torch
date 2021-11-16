from os import listdir
from os.path import isfile
from os.path import join
from os.path import isdir
from os.path import abspath
import os
from torchvision.datasets.video_utils import VideoClips

from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_video
import torchvision as tv
import csv
from torch import nn
import json
from torchvision.transforms import ToTensor

class MinMaxScaler(object):
    """
    Transforms each channel to the range [0, 1].
    """
    def __init__(self, min_val=0, max_val=254):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, tensor):
        return (tensor - self.min_val) / (self.max_val - self.min_val)

class VideoGrayScaler(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.grayscale = tv.transforms.Grayscale(num_output_channels=1)
        
    def forward(self, video):
        # shape = channels, time, width, height
        video = self.grayscale(video.swapaxes(-4, -3).swapaxes(-2, -1))
        video = video.swapaxes(-4, -3).swapaxes(-2, -1)
        # print(video.shape)
        return video
    
class VideoLoader(Dataset):
    
    def __init__(self, video_path, transform=None, num_frames=None):
        self.num_frames = num_frames
        self.transform = transform
        
        self.labels = [name for name in listdir(video_path) if isdir(join(video_path, name))]
        self.videos = []
        for label in self.labels:
            self.videos.extend([(label, abspath(join(video_path, label, f)), f) for f in listdir(join(video_path, label)) if isfile(join(video_path, label, f))])
            
        self.cache = {}
        
    def get_filenames(self):
        return [self.videos[i][1].split('/')[-1].split('.')[0].lower().replace('_clean', '') for i in range(len(self.videos))]
        
    def get_labels(self):
        return [self.videos[i][0] for i in range(len(self.videos))]
    
    def __getitem__(self, index):
        #print('index: {}'.format(index))
        
        if index in self.cache:
            return self.cache[index]
        
        label = self.videos[index][0]
        filename = self.videos[index][2]
        video, _, info = read_video(self.videos[index][1])
        # print(info)
        video = torch.swapaxes(video, 1, 3)
        
        # print('length', len(video))
        if self.num_frames:
            video = video[-self.num_frames:]
            video = np.stack([self.transform(f) for f in video], axis=0)
            
            if len(video) < self.num_frames:
                padding = torch.zeros(self.num_frames - len(video), video.shape[1], video.shape[2], video.shape[3])
                video = torch.cat((video, padding))
            
        video = video.swapaxes(0, 1).swapaxes(2, 3)
        
#         if self.transform:
#             video = self.transform(video)
        
        self.cache[index] = (label, video)
            
        return label, video
        
    def __len__(self):
        return len(self.videos)
    
class VideoClipLoader(Dataset):
    
    def __init__(self, video_path, meta_file, clip_length_in_frames=20, frame_rate=20, frames_between_clips=None, transform=None):
        video_to_label = {}
        with open(meta_file, 'r') as csv_in:
            csv_reader = csv.DictReader(csv_in)
            for row in csv_reader:
                video_to_label[row['Filename']] = row['Label']
                
        self.transform = transform
        self.names = [name for name in listdir(video_path) if name.split('.')[0] in video_to_label]
        
        self.labels = [video_to_label[name.split('.')[0]] for name in self.names]
        
        self.label_vocab = {lbl: i for i, lbl in enumerate(set(self.labels))}
        
        self.videos = [(video_to_label[name.split('.')[0]], abspath(join(video_path, name))) for name in self.names]
            
        if not frames_between_clips:
            frames_between_clips = clip_length_in_frames
            
        vc = VideoClips([path for _, path, _ in self.videos],
                        clip_length_in_frames=clip_length_in_frames,
                        frame_rate=frame_rate,
                       frames_between_clips=frames_between_clips)
        self.clips = []
        self.video_idx = []
        
        if os.path.exists('clip_cache.pt'):
            self.clips = torch.load(open('clip_cache.pt', 'rb'))
            self.video_idx = torch.load(open('video_idx_cache.pt', 'rb'))
        else:
            for i in tqdm(range(vc.num_clips())):
                try:
                    clip, _, _, vid_idx = vc.get_clip(i)
                    clip = clip.swapaxes(1, 3).swapaxes(2, 3)
                    if self.transform:
                        clip = self.transform(clip)
                    clip = clip.swapaxes(0, 1)
                    self.clips.append((self.label_vocab[self.videos[vid_idx][0]], clip))
                    self.video_idx.append(vid_idx)
                except Exception as e:
                    pass


            torch.save(self.clips, open('clip_cache.pt', 'wb+'))
            torch.save(self.video_idx, open('video_idx_cache.pt', 'wb+'))
        
    def get_video_labels(self):
        return [self.videos[i][0] for i in range(len(self.videos))]
        
    def get_labels(self):
        return [self.clips[i][0] for i in range(len(self.clips))]
    
    def label_vocab_size(self):
        return len(self.label_vocab)
    
    def __getitem__(self, index):
        return self.clips[index]
        
    def __len__(self):
        return len(self.clips)
    
class VideoClipLoaderBAMC(Dataset):
    
    def __init__(self, video_path, sparse_model=None, device=None, transform=None, frame_rate=20, num_frames=1, augment_transform=None):
        self.transform = transform
        self.augment_transform = augment_transform
        
        self.labels = [name for name in listdir(video_path) if isdir(join(video_path, name))]
        self.label_vocab = {lbl: i for i, lbl in enumerate(set(self.labels))}
        
        self.videos = []
        for label in self.labels:
            self.videos.extend([(label, abspath(join(video_path, label, f))) for f in listdir(join(video_path, label)) if isfile(join(video_path, label, f))])
            
        self.cache = {}
        
        self.sparse_model = sparse_model
            
        vc = VideoClips([path for label, path in self.videos],
                        clip_length_in_frames=5,
                        frame_rate=1,
                       frames_between_clips=5)
        self.clips = []
        self.video_idx = []
        
        if os.path.exists('clip_cache_bamc.pt'):
            self.clips = torch.load(open('clip_cache_bamc.pt', 'rb'))
            self.video_idx = torch.load(open('video_idx_cache_bamc.pt', 'rb'))
        else:
            for i in tqdm(range(vc.num_clips())):
                try:
                    clip, _, _, vid_idx = vc.get_clip(i)
                    clip = clip.swapaxes(1, 3).swapaxes(0, 1).swapaxes(2, 3).to(torch.float)
                    if self.transform:
                        clip = self.transform(clip)
                        
                    if sparse_model:
                        with torch.no_grad():
                            clip = clip.unsqueeze(0).to(device)
                            u_init = torch.zeros([1, sparse_model.out_channels] + sparse_model.get_output_shape(clip))
                            clip, _ = self.sparse_model(clip, u_init)
                            clip = clip.squeeze(0).detach()

                    self.clips.append((self.videos[vid_idx][0], clip, self.videos[vid_idx][1]))
                    self.video_idx.append(vid_idx)
                except Exception as e:
                    print(e)
                    raise e


            torch.save(self.clips, open('clip_cache_bamc.pt', 'wb+'))
            torch.save(self.video_idx, open('video_idx_cache_bamc.pt', 'wb+'))
        
    def get_filenames(self):
        return [self.videos[i][1].split('/')[-1].split('.')[0].lower().replace('_clean', '') for i in self.video_idx]
        
    def get_labels(self):
        return [self.clips[i][0] for i in range(len(self.clips))]
    
    def label_vocab_size(self):
        return len(set(self.labels))
    
    def __getitem__(self, index):
        #print('index: {}'.format(index))
        lbl, clip, vid_f = self.clips[index]
        clip = self.augment_transform(clip)
        
        return lbl, clip, vid_f
        
    def __len__(self):
        return len(self.clips)
    
class VideoFrameLoader(Dataset):
    
    def __init__(self, video_path, sparse_model=None, device=None, transform=None, num_frames=None):
        self.num_frames = num_frames
        self.transform = transform
        
        self.labels = [name for name in listdir(video_path) if isdir(join(video_path, name))]
        self.videos = []
        for label in self.labels:
            self.videos.extend([(label, abspath(join(video_path, label, f)), f) for f in listdir(join(video_path, label)) if isfile(join(video_path, label, f))])
            
        self.sparse_model = sparse_model
        self.device = device
            
        self.cache = {}
        
    def get_filenames(self):
        return [self.videos[i][1].split('/')[-1].split('.')[0].lower().replace('_clean', '') for i in range(len(self.videos))]
        
    def get_labels(self):
        return [self.videos[i][0] for i in range(len(self.videos))]
    
    def __getitem__(self, index):
        #print('index: {}'.format(index))
        
        if index in self.cache:
            return self.cache[index]
        
        label = self.videos[index][0]
        filename = self.videos[index][2]
        video, _, info = read_video(self.videos[index][1])
        # print(info)
        video = torch.swapaxes(video, 1, 3)
        
        # print('length', len(video))
        if self.num_frames:
            video = video[-self.num_frames:]
            
            if len(video) < self.num_frames:
                padding = torch.zeros(self.num_frames - len(video), video.shape[1], video.shape[2], video.shape[3])
                video = torch.cat((video, padding))
            
        video = video.swapaxes(0, 1).swapaxes(2, 3)
        
        if self.transform:
            video = self.transform(video)
            
        if self.sparse_model:
            with torch.no_grad():
                video = video.swapaxes(0, 1).unsqueeze(1).to(self.device)
                
                new_vids = []
                for i in range(0, len(video), 5):
                    try:
                        sub_vid = video[i:i+5]
                    except IndexError:
                        sub_vid = video[i:]
                    u_init = torch.zeros([sub_vid.size(0), self.sparse_model.out_channels] + self.sparse_model.get_output_shape(sub_vid))
                    sub_vid, _ = self.sparse_model(sub_vid, u_init)
                    sub_vid = sub_vid.squeeze(2).detach()
                    new_vids.append(sub_vid)
                video = torch.cat(new_vids, dim=0).to('cpu')
        
        self.cache[index] = (label, video, filename)
            
        return label, video, filename
        
    def __len__(self):
        return len(self.videos)

class YoloClipLoader(Dataset):
    
    def __init__(self, yolo_output_path, num_frames=5, frames_between_clips=None,
                 transform=None, augment_transform=None, sparse_model=None, device=None):
        if (num_frames % 2) == 0:
            raise ValueError("Num Frames must be an odd number, so we can extract a clip centered on each detected region")
        
        clip_cache_file = 'clip_cache.pt'
        
        self.num_frames = num_frames
        if frames_between_clips is None:
            self.frames_between_clips = num_frames
        else:
            self.frames_between_clips = frames_between_clips

        self.transform = transform
        self.augment_transform = augment_transform
         
        self.labels = [name for name in listdir(yolo_output_path) if isdir(join(yolo_output_path, name))]
        self.clips = []
        if os.path.exists(clip_cache_file):
            self.clips = torch.load(open(clip_cache_file, 'rb'))
        else:
            for label in self.labels:
                print("Processing videos in category: {}".format(label))
                videos = list(listdir(join(yolo_output_path, label)))
                for vi in tqdm(range(len(videos))):
                    video = videos[vi]
                    with open(abspath(join(yolo_output_path, label, video, 'result.json'))) as fin:
                        results = json.load(fin)
                        max_frame = len(results)

                        for i in range((num_frames-1)//2, max_frame - (num_frames-1)//2 - 1, self.frames_between_clips):
                        # for frame in results:
                            frame = results[i]
                            # print('loading frame:', i, frame['frame_id'])
                            frame_start = int(frame['frame_id']) - self.num_frames//2
                            frames = [abspath(join(yolo_output_path, label, video, 'frame{}.png'.format(frame_start+fid)))
                                      for fid in range(num_frames)]
                            # print(frames)
                            frames = torch.stack([ToTensor()(Image.open(f).convert("RGB")) for f in frames]).swapaxes(0, 1)

                            for region in frame['objects']:
                                # print(region)
                                if region['name'] != "Pleural_Line":
                                    continue

                                center_x = region['relative_coordinates']["center_x"] * 1920
                                center_y = region['relative_coordinates']['center_y'] * 1080

                                # width = region['relative_coordinates']['width'] * 1920
                                # height = region['relative_coordinates']['height'] * 1080
                                width=400
                                height=400

                                lower_y = round(center_y - height / 2)
                                upper_y = round(center_y + height / 2)
                                lower_x = round(center_x - width / 2)
                                upper_x = round(center_x + width / 2)

                                final_clip = frames[:, :, lower_y:upper_y, lower_x:upper_x]

                                if self.transform:
                                    final_clip = self.transform(final_clip)

                                if sparse_model:
                                    with torch.no_grad():
                                        final_clip = final_clip.unsqueeze(0).to(device)
                                        u_init = torch.zeros([1, sparse_model.out_channels] + sparse_model.get_output_shape(final_clip))
                                        final_clip, _ = sparse_model(final_clip, u_init)
                                        final_clip = final_clip.squeeze(0).detach().cpu()

                                self.clips.append((label, final_clip, video))

            torch.save(self.clips, open(clip_cache_file, 'wb+'))
                            

            
    def get_labels(self):
        return [self.clips[i][0] for i in range(len(self.clips))]
    
    def get_filenames(self):
        return [self.clips[i][2] for i in range(len(self.clips))]
    
    def __getitem__(self, index):        
        label = self.clips[index][0]
        video = self.clips[index][1]
        filename = self.clips[index][2]
        
        if self.augment_transform:
            video = self.augment_transform(video)
            
        return label, video, filename
        
    def __len__(self):
        return len(self.clips)
    

if __name__ == "__main__":
    video_path = "/shared_data/bamc_data/"
    
    transforms = tv.transforms.Compose([VideoGrayScaler()])

    # dataset = VideoLoader(video_path, transform=transforms, num_frames=60)
    dataset = VideoClipLoader(video_path, transform=transforms, clip_length_in_frames=20)
    #for data in dataset:
    #    print(data[0], data[1].shape)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True)
    
    for data in loader:
        print(data[0], data[1].shape, data[2])
        #print(data)