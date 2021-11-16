import numpy as np
import torchvision
import torch
from sklearn.model_selection import train_test_split, GroupShuffleSplit, LeaveOneGroupOut, LeaveOneOut, StratifiedGroupKFold, StratifiedKFold
from utils.video_loader import MinMaxScaler, VideoGrayScaler
from utils.video_loader import VideoLoader
from utils.video_loader import VideoClipLoader, VideoClipLoaderBAMC, VideoFrameLoader, YoloClipLoader
import csv

def load_balls_data(batch_size):
    
    with open('ball_videos.npy', 'rb') as fin:
        ball_videos = torch.tensor(np.load(fin)).float()

    batch_size = batch_size
    train_loader = torch.utils.data.DataLoader(ball_videos,
                                               batch_size=batch_size,
                                               shuffle=True)

    return train_loader

def load_bamc_data(batch_size, train_ratio, sparse_model=None, device=None, seed=None):   
    video_path = "/shared_data/bamc_data_scale_cropped/"
    
    scale = 0.2
    
    base_width = 1920
    base_height = 1080
    
    cropped_width = round(140/320 * base_width)
    cropped_height = round(140/180 * base_height)
    
    width = round(cropped_width * scale)
    height = round(cropped_height * scale)
    
    video_to_participant = {}
    with open('/shared_data/bamc_data/bamc_video_info.csv', 'r') as csv_in:
        reader = csv.DictReader(csv_in)
        for row in reader:
            key = row['Filename'].split('.')[0].lower().replace('_clean', '')
            if key == '37 (mislabeled as 38)':
                key = '37'
            video_to_participant[key] = row['Participant_id']
            
    
#     transforms = torchvision.transforms.Compose([VideoGrayScaler(),
#                                                  torchvision.transforms.Resize(size=(width, height)), 
#                                                  MinMaxScaler(0, 255),
#                                                 torchvision.transforms.RandomRotation(15)])
#     transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(width, height)), MinMaxScaler(0, 255), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), torchvision.transforms.RandomRotation(15)])
    transforms = torchvision.transforms.Compose([VideoGrayScaler(),

                                             MinMaxScaler(0, 255),

                                             torchvision.transforms.Normalize((0.184914231300354,), (0.11940956115722656,)),

                                             torchvision.transforms.Resize(size=(200, 350))

                                            ])
    
    dataset = VideoFrameLoader(video_path, sparse_model=sparse_model, device=device, transform=transforms, num_frames=60)
    
    targets = dataset.get_labels()
    
    if train_ratio == 1.0:
        train_idx = np.arange(len(targets))
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               # shuffle=True,
                                               sampler=train_sampler)
        test_loader = None
    else:
        gss = StratifiedGroupKFold(n_splits=5)#, train_size=train_ratio, random_state=seed)

        groups = [video_to_participant[v] for v in dataset.get_filenames()]
        
        return gss.split(np.arange(len(targets)), targets, groups), dataset
#         gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)

#         groups = [video_to_participant[v] for v in dataset.get_filenames()]

#         train_idx, test_idx = next(gss.split(np.arange(len(targets)), targets, groups))

#     #     train_idx, test_idx = train_test_split(np.arange(len(targets)), test_size=0.2, shuffle=True, stratify=targets)

#         train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
#         test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

#         train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                                    # shuffle=True,
#                                                    sampler=train_sampler)

#         test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                                         # shuffle=True,
#                                                         sampler=test_sampler)

    return train_loader, test_loader

def load_covid_data(batch_size, clip_length_in_frames=10, frame_rate=20, train_ratio=0.8):   
    video_path = "/home/dwh48@drexel.edu/covid19_ultrasound/data/pocus_videos/convex"
    meta_path = "/home/dwh48@drexel.edu/covid19_ultrasound/data/dataset_metadata.csv"
    # video_path = "/home/cm3786@drexel.edu/Projects/covid19_ultrasound/data/pocus_videos/pneumonia-viral"
    
    scale = 0.5
    
    base_width = 1920
    base_height = 1080
    
    cropped_width = round(140/320 * base_width)
    cropped_height = round(140/180 * base_height)
    
    #width = round(cropped_width * scale)
    #height = round(cropped_height * scale)
    
    width = 128
    height = 128
    
    transforms = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                                 #torchvision.transforms.Resize(size=(base_width, base_height)),
                                                 #torchvision.transforms.CenterCrop(size=(cropped_height, cropped_width)),
                                                 torchvision.transforms.Resize(size=(width, height)), 
                                                 MinMaxScaler(0, 255)])
    dataset = VideoClipLoader(video_path, meta_path, transform=transforms,
                              clip_length_in_frames=clip_length_in_frames,
                              frame_rate=frame_rate)
    
    targets = dataset.get_video_labels()

    train_vidx, test_vidx = train_test_split(np.arange(len(targets)), test_size=0.2, shuffle=True, stratify=targets)
    
    train_vidx = set(train_vidx)
    test_vidx = set(test_vidx)
    
    train_cidx = [i for i in range(len(dataset)) if dataset.video_idx[i] in train_vidx]
    test_cidx = [i for i in range(len(dataset)) if dataset.video_idx[i] in test_vidx]
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_cidx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_cidx)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              sampler=test_sampler)

    return train_loader, test_loader

def load_bamc_clips(batch_size, train_ratio, sparse_model, device, num_frames=1, seed=None):   
    video_path = "/shared_data/YOLO_PL_Model_Results/"
    
    scale = 0.2
    
    base_width = 1920
    base_height = 1080
    
    cropped_width = 224
    cropped_height = 224
    
    width = 256
    height = 256
    
    video_to_participant = {}
    with open('/shared_data/bamc_data/bamc_video_info.csv', 'r') as csv_in:
        reader = csv.DictReader(csv_in)
        for row in reader:
            key = row['Filename'].split('.')[0].lower().replace('_clean', '')
            if key == '37 (mislabeled as 38)':
                key = '37'
            video_to_participant[key] = row['Participant_id']
            
    
#     transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(width, height)), torchvision.transforms.CenterCrop(size=(cropped_height, cropped_width)), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), torchvision.transforms.RandomRotation(15)])
    transforms = torchvision.transforms.Compose([VideoGrayScaler(),
                                             MinMaxScaler(0, 255),
                                             torchvision.transforms.Normalize((0.184914231300354,), (0.11940956115722656,)),
                                             # BamcPreprocessor(),
                                             torchvision.transforms.Resize(size=(200, 350))
                                            ])
    augment_transform = torchvision.transforms.RandomRotation(15)
    dataset = VideoClipLoaderBAMC(video_path, sparse_model=sparse_model, device=device, transform=transforms, frame_rate=20, num_frames=num_frames, augment_transform=augment_transform)
    
    targets = dataset.get_labels()
    
    if train_ratio is None:
        gss = LeaveOneGroupOut()

        groups = [v for v in dataset.get_filenames()]
        
        return gss.split(np.arange(len(targets)), targets, groups), dataset
#         gss = LeaveOneOut()
        
#         return gss.split(np.arange(len(targets)), targets), dataset
    elif train_ratio == 1.0:
        train_idx = np.arange(len(targets))
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               # shuffle=True,
                                               sampler=train_sampler)
        test_loader = None
    else:
        gss = StratifiedGroupKFold(n_splits=5)#, train_size=train_ratio, random_state=seed)

        groups = [video_to_participant[v] for v in dataset.get_filenames()]
        
        return gss.split(np.arange(len(targets)), targets, groups), dataset
#           gss = StratifiedKFold(n_splits=5)#, train_size=train_ratio, random_state=seed)
        
#           return gss.split(np.arange(len(targets)), targets), dataset

#         train_idx, test_idx = next(gss.split(np.arange(len(targets)), targets, groups))

#     #     train_idx, test_idx = train_test_split(np.arange(len(targets)), test_size=0.2, shuffle=True, stratify=targets)

#         train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
#         test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

#         train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                                    # shuffle=True,
#                                                    sampler=train_sampler)

#         test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                                         # shuffle=True,
#                                                         sampler=test_sampler)

    return train_loader, test_loader

def load_yolo_clips(batch_size, mode, device, n_splits=None, sparse_model=None):   
    video_path = "/shared_data/YOLO_Updated_PL_Model_Results/"

    video_to_participant = {}
    with open('/shared_data/bamc_data/bamc_video_info.csv', 'r') as csv_in:
        reader = csv.DictReader(csv_in)
        for row in reader:
            key = row['Filename'].split('.')[0].lower().replace('_clean', '')
            if key == '37 (mislabeled as 38)':
                key = '37'
            video_to_participant[key] = row['Participant_id']
            
    
    transforms = torchvision.transforms.Compose(
    [VideoGrayScaler(),
     torchvision.transforms.Normalize((0.2592,), (0.1251,)),
    ])
    augment_transforms = torchvision.transforms.Compose(
    [torchvision.transforms.RandomRotation(45),
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.CenterCrop((100, 200))
    ])
    dataset = YoloClipLoader(video_path, transform=transforms, augment_transform=augment_transforms, sparse_model=sparse_model, device=device)
    
    targets = dataset.get_labels()
    
    if mode == 'leave_one_out':
        gss = LeaveOneGroupOut()

#         groups = [v for v in dataset.get_filenames()]
        groups = [video_to_participant[v.lower().replace('_clean', '')] for v in dataset.get_filenames()]
        
        return gss.split(np.arange(len(targets)), targets, groups), dataset
    elif mode == 'all_train':
        train_idx = np.arange(len(targets))
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
        test_loader = None
        
        return train_loader, test_loader
    elif mode == 'k_fold':
        gss = StratifiedGroupKFold(n_splits=n_splits)

        groups = [video_to_participant[v.lower().replace('_clean', '')] for v in dataset.get_filenames()]
        
        return gss.split(np.arange(len(targets)), targets, groups), dataset
    else:
        return None