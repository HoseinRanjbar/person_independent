#############################################
#                                           #
# Load sequential data from PHOENIX-2014    #
#                                           #
#############################################

from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import math

import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Sampler
from torchvision import transforms, utils
from tools.indexs_list import idxs
from collections import defaultdict



#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def collate_fn(data, fixed_padding=None, pad_index=1232):
    """Creates mini-batch tensors w/ same length sequences by performing padding to the sequecenses.
    We should build a custom collate_fn to merge sequences w/ padding (not supported in default).
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding), else pad
    all Sequences to a fixed length.

    Returns:
        hand_seqs: torch tensor of shape (batch_size, padded_length).
        hand_lengths: list of length (batch_size); 
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); 
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); 
    """

    def pad(sequences, t):
        lengths = [len(seq) for seq in sequences]

        #For sequence of images
        if(t=='source'):
            #Retrieve shape of single sequence
            #(seq_length, channels, n_h, n_w)
            seq_shape = sequences[0].shape
            if(fixed_padding):
                padded_seqs = fixed_padding
                padded_seqs = torch.zeros(len(sequences), fixed_padding, seq_shape[1], seq_shape[2], seq_shape[3]).type_as(sequences[0])
            else:
                padded_seqs = torch.zeros(len(sequences), max(lengths), seq_shape[1], seq_shape[2], seq_shape[3]).type_as(sequences[0])

        #For sequence of words
        elif(t=='target'):
            # Just convert the list of target words to a tensor directly
            padded_seqs = torch.tensor(sequences, dtype=torch.long)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]

        return padded_seqs, lengths

    src_seqs = []
    trg_seqs = []
    right_hands = []
    left_hands = []

    for element in data:
        src_seqs.append(element['images'])
        trg_seqs.append(element['translation'])

        right_hands.append(element['right_hands'])

    #pad sequences
    src_seqs, src_lengths = pad(src_seqs, 'source')
    # Convert target sequences to tensor (no padding needed)
    trg_seqs = torch.tensor(trg_seqs, dtype=torch.long).view(-1, 1)

    #pad hand sequences
    if(type(right_hands[0]) != type(None)):
        hand_seqs, hand_lengths = pad(right_hands, 'source')
    else:
        hand_seqs = None
        hand_lengths = None

    return src_seqs, src_lengths, trg_seqs, hand_seqs, hand_lengths


class ClassBalancedBatchSampler(Sampler):
    """
    Builds batches with:
        - classes_per_batch classes
        - samples_per_class samples per class
      so batch_size = classes_per_batch * samples_per_class.

    This ensures multiple samples per class in each batch,
    which is great for supervised contrastive learning.
    """
    def __init__(self, labels, batch_size, samples_per_class=2, drop_last=True,
                 debug=False, max_debug_batches=3):
        """
        labels: list/array of int class labels, len = len(dataset)
        batch_size: total batch size (must be divisible by samples_per_class)
        samples_per_class: K, number of samples of each class per batch
        """
        self.labels = list(labels)
        self.samples_per_class = samples_per_class
        self.drop_last = drop_last

        # DEBUG flags
        self.debug = debug
        self.max_debug_batches = max_debug_batches
        self._debug_batch_count = 0

        print('batch_size:{}, sample_per_class:{}'.format(batch_size,samples_per_class))
        assert batch_size % samples_per_class == 0, \
            "batch_size must be divisible by samples_per_class"

        self.classes_per_batch = batch_size // samples_per_class

        # class_label -> [indices...]
        self.class_to_indices = defaultdict(list)
        for idx, y in enumerate(self.labels):
            self.class_to_indices[y].append(idx)

        self.classes = [c for c, idxs in self.class_to_indices.items() if len(idxs) > 0]
        # labels = [lookup_table.get(c, -1) for c in transformed_dataset.annotations['class'].values]


    def __iter__(self):
        # Fresh copy for this epoch
        class_to_pool = {c: idxs.copy() for c, idxs in self.class_to_indices.items()}

        # Shuffle indices inside each class
        for c in self.classes:
            random.shuffle(class_to_pool[c])

        while True:
            # Classes that still have available indices
            available_classes = [c for c in self.classes if len(class_to_pool[c]) > 0]
            if len(available_classes) < self.classes_per_batch:
                # Not enough classes to form a full batch
                break

            # Choose which classes go into this batch
            chosen_classes = random.sample(available_classes, self.classes_per_batch)

            batch = []
            for c in chosen_classes:
                # Refill if not enough left in this class
                if len(class_to_pool[c]) < self.samples_per_class:
                    original = self.class_to_indices[c]
                    extra_needed = self.samples_per_class - len(class_to_pool[c])
                    refill = random.sample(original, extra_needed)
                    class_to_pool[c].extend(refill)

                # Take K indices for this class
                for _ in range(self.samples_per_class):
                    batch.append(class_to_pool[c].pop())

            # >>> DEBUG: print first few batches
            if self.debug and self._debug_batch_count < self.max_debug_batches:
                batch_labels = [self.labels[i] for i in batch]
                print("\n[Sampler DEBUG] Batch", self._debug_batch_count)
                print("  chosen_classes:", chosen_classes)
                print("  batch indices:", batch)
                print("  batch labels:", batch_labels)

                # Count how many times each label appears
                from collections import Counter
                counts = Counter(batch_labels)
                print("  label counts:", counts)
                self._debug_batch_count += 1

            yield batch

    def __len__(self):
        # Rough estimate
        return math.floor(len(self.labels) / (self.classes_per_batch * self.samples_per_class))


#From abstract function Dataset
class ISLRDataset(Dataset):
    """Sequential Sign language images dataset."""

    def __init__(self, csv_file, root_dir, lookup_table, random_drop, uniform_drop, istrain,remove_bg = None, transform=None,rescale=224, sos_index=1, eos_index=2, unk_index=0, fixed_padding=None, hand_dir=None, hand_transform=None, channels=3):

        #Get data
        #self.annotations = pd.read_csv(csv_file)
        self.annotations = csv_file
        # 1) Remove any rows that point to .npy files (like pose.npy)
        self.annotations = self.annotations[
            ~self.annotations['video_path'].str.endswith('.npy')]

        # 2) (optional but nice) keep only mp4 / avi / etc.
        self.annotations = self.annotations[
            self.annotations['video_path'].str.endswith(('.mp4', '.avi', '.mov'))]

        # 3) Reset index so __len__ and __getitem__ stay consistent
        self.annotations = self.annotations.reset_index(drop=True)

        self.root_dir = root_dir
        self.lookup_table = lookup_table
        self.remove_bg= remove_bg
        self.hand_dir = hand_dir
        self.random_drop = random_drop
        self.uniform_drop = uniform_drop
        self.transform = transform
        self.hand_transform = hand_transform
        self.istrain = istrain
        self.rescale = rescale

        self.channels = channels

        #index used for eos token and unk
        self.eos_index = eos_index
        self.unk_index = unk_index
        self.sos_index = sos_index


    def __len__(self):
        #Return size of dataset
        return len(self.annotations)

    def __getitem__(self, idx):
        #global trsf_images
        #Retrieve the name id of sequence from csv annotations
        name = self.annotations.iloc[idx]['video_path']

        name = name.replace('\\', '/')

        # 2. Join with root_dir
        video_path = os.path.join(self.root_dir, name)

        # 3. (Optional but good) Normalize the path
        video_path = os.path.normpath(video_path)

        # Create a VideoCapture object
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       
        if self.istrain:
            indexs = idxs(frame_count,random_drop=self.random_drop,uniform_drop=self.uniform_drop)
            seq_length = len(indexs)
        else:
            indexs = idxs(frame_count,random_drop=None,uniform_drop= self.uniform_drop)
            seq_length = len(indexs)

        trsf_images = torch.zeros((seq_length, self.channels, self.rescale, self.rescale))

        #Get hand cropped image list if exists
        if(self.hand_dir):
            hand_path = os.path.join(self.hand_dir, name)
            hand_images = torch.zeros((seq_length, self.channels, 112, 112))
        else:
            hand_images = None

        #Save the images of seq
        i=0
        j=0
        # Loop through the video frames
        while True:

            # Capture frame-by-frame
            ret, frame = cap.read()        
            #image=cv2.imread(img_name)
                # If no frame is returned, break the loop (end of the video)
            if not ret:
                break

            if i in indexs:

                # Convert the frame from BGR (OpenCV format) to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image= cv2.resize(frame,(224,224))
                #NOTE: some images got shape of (260, 220, 4)
                if(image.shape[2] == self.channels):
                    trsf_images[j] = self.transform(image)
                else:
                    trsf_images[j] = self.transform(image[:, :, :self.channels])
                j+=1

            i+=1

        cap.release()
        #Retrive the translation (ground truth text translation) from csv annotations
        sign = self.annotations.iloc[idx]['class']

        # Convert the ground truth label to numeric using the lookup table
        label = self.lookup_table.get(sign, -1)  # Default to -1 if the class is not in the lookup table
        label = torch.tensor([label], dtype=torch.long).squeeze()

        #NOTE: full frame seq and hand seq should be with the same seq length
        #sample = {'images': trsf_images, 'right_hands':hand_images, 'translation': trans}
        return {'images': trsf_images, 'right_hands':hand_images, 'translation': label}
        #return sample


# Helper function to show a batch
def show_batch(sample_batched):
    """Show sequence of images with translation for a batch of samples."""

    images_batch, images_length, trans_batch, trans_length = \
            sample_batched
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    #Show only one sequence of the batch
    grid = utils.make_grid(images_batch[0, :images_length[0]])
    grid = grid.numpy()
    return np.transpose(grid, (1,2,0))


#Use this to subtract mean from each pixel measured from PHOENIX-T dataset
#Note: means has been subtracted from 227x227 images, this has been provided by camgoz
class SubtractMeans(object):
    def __init__(self, path, rescale):
        #NOTE: Newest np versions default value allow_pickle=False
        self.mean = np.load(path, allow_pickle=True)
        self.mean = self.mean.astype('uint8')
        self.rescale = rescale

    def __call__(self, image):

        #No need to resize (take long time..)
        #image = cv2.resize(image,(self.mean.shape[0], self.mean.shape[1]))
        assert image.shape == self.mean.shape
        image -= self.mean
        #image = cv2.resize(image,(self.rescale, self.rescale))

        return image


def loader(csv_file, root_dir, lookup_table, rescale, batch_size, samples_per_class, num_workers, random_drop, uniform_drop, show_sample, debug = None, remove_bg = None,  local_rank=0, istrain=False, mean_path='FulFrame_Mean_Image_227x227.npy', fixed_padding=None, hand_dir=None, data_stats=None, hand_stats=None, channels=3):

    #Note: when using random cropping, this with reshape images with randomCrop size instead of rescale
    if(istrain):

        if(data_stats):
            trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.Resize((rescale, rescale)),
                transforms.ToTensor()
                #transforms.Normalize(mean=data_stats['mean'], std=data_stats['std'])
                ])

        
        else:
            trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.Resize((rescale, rescale)),
                transforms.ToTensor()
                #Imagenet std and mean
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            

        if(hand_stats):
            hand_trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomAffine(10),
                    transforms.Resize((112, 112)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=hand_stats['mean'], std=hand_stats['std'])
                    ])
        else:
            hand_trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomAffine(10),
                    transforms.Resize((112, 112)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                

    else:

        if(data_stats):
            trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((rescale, rescale)),
                transforms.ToTensor()
                #transforms.Normalize(mean=data_stats['mean'], std=data_stats['std'])
                ])
            

        else:
             trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((rescale, rescale)),
                transforms.ToTensor()
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])


        if(hand_stats):
            hand_trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=hand_stats['mean'], std=hand_stats['std'])
                    ])
        else:
            hand_trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

    ##Iterate through the dataset and apply data transformation on the fly

    #Apply data augmentation to avoid overfitting
    transformed_dataset = ISLRDataset(csv_file=csv_file,
                                            root_dir=root_dir,
                                            lookup_table=lookup_table,
                                            remove_bg=remove_bg,
                                            random_drop=random_drop,
                                            uniform_drop=uniform_drop,
                                            transform=trans,
                                            rescale=rescale,
                                            istrain=istrain,
                                            hand_dir=hand_dir,
                                            hand_transform=hand_trans,
                                            channels = channels
                                            )

    size = len(transformed_dataset)

    # Decide whether to use DistributedSampler or normal shuffling
    use_distributed = dist.is_available() and dist.is_initialized()

    batch_sampler = None

    if use_distributed:
        world_size = dist.get_world_size()
        sampler = DistributedSampler(
            transformed_dataset,
            rank=local_rank,
            num_replicas=world_size,
            shuffle=True,  # let sampler handle shuffling
        )
        shuffle = False  # cannot use shuffle=True when a sampler is set
    else:
        # >>> NEW: for training, use class-balanced batch sampler
        if istrain:
            labels = transformed_dataset.annotations['class'].values  # column name from your CSV
            # samples_per_class = 2   # or 3/4 if batch_size is large enough

            batch_sampler = ClassBalancedBatchSampler(
                labels=labels,
                batch_size=batch_size,
                samples_per_class=samples_per_class,
                drop_last=True,
                debug=debug
            )
            sampler = None
            shuffle = False   # DataLoader must NOT shuffle when using batch_sampler
        else:
            # validation / test: normal sequential or shuffled sampling
            sampler = None
            batch_sampler = None
            shuffle = True

    # Now build DataLoader
    if batch_sampler is not None:
        dataloader = DataLoader(
            transformed_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    else:
        dataloader = DataLoader(
            transformed_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )


    #Show a sample of the dataset
    if(show_sample and istrain):
        for i_batch, sample_batched in enumerate(dataloader):
            #plt.figure()
            img = show_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.imshow(img)
            #plt.show()
            plt.savefig('data_sample.png')
            break

    return dataloader, size
