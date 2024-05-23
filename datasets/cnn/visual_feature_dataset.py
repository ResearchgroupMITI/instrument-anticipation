"""dataset loader for visual feature module"""

import os
import pickle
import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms

class VisualDataset:
    def __init__(self, hparams):
        self.hparams = hparams
        self.data = self.create_data_sets()


    def create_data_sets(self):
        datasets = {}
        for split in ["train", "val", "test", "all"]:
            datasets[split] = Visual_Helper(self.hparams, split)
        return datasets


    @staticmethod
    def add_dataset_specific_args(parser):
        dataset_specific_args = parser.add_argument_group(
            title='dataset specific args options')
        dataset_specific_args.add_argument("--num_features",
                                            default=53,
                                            type=int)
        dataset_specific_args.add_argument("--augmentation_type", type=str, default="tecno")
        dataset_specific_args.add_argument("--num_instruments", type=int, default=10)
        return parser


class Visual_Helper(Dataset):
    def __init__(self, hparams, mode):
        self.hparams = hparams
        self.img_dir = os.path.join(self.hparams.data_root, 'images')
        self.img_tensor_dir = os.path.join(self.hparams.data_root, 'image_tensors')
        self.anno_dir = os.path.join(self.hparams.data_root, 'anno')
        self.split_dir = os.path.join(self.hparams.data_root, 'data_split')
        self.pregenerated_list_path = os.path.join(self.hparams.data_root, "pregenerated")
        self.mode = mode

        self.transformations = self.get_transformations()

        train_ids, val_ids, test_ids, all_video_ids = self.load_split_ids()

        #names of the instruments annotated for left and right trocar
        self.ins_names_trocar = ['Idle', 'Grasper', 'PE-Grasper', 'Clipper', 'Scissors', 'Irrigator', 'Retrieval Bag', 'Drainage', 'PE-Rod', 'Resting']

        #list of the features we want to use for the ML model
        self.ins_names = ['PE Active', 'Grasper', 'Palpation Probe', 'Flusher', 'Clipper', 'Scissor', 'Trocar', 'PE Grasper', 'PE Rod']
        phases_names = ['Phase Trocar Insertion', 'Phase Preparation', 'Phase Calot Triangle Dissection', 'Phase Clipping', 'Phase Dissection',
                'Phase Haemostasis', 'Phase Packaging', 'Phase Retraction', 'Phase Instrument ReInsertion', 'Phase Haemostasis 2', 'Phase Trocar Removal']
        action_names = ['Action Trocar Insertion', 'Action Preparation', 'Action Calot Triangle Dissection', 'Action Dissection', 'Action Haemostasis',
                        'Action Cleaning', 'Action Packaging', 'Action Retraction', 'Action Instrument ReInsertion', 'Action Trocar Removal']
        gen_surg_actphas_names = ['Idle', 'Specimen Bag', 'Drainage']
        ins_left_trocar = ['LT - ' + ins for ins in self.ins_names_trocar]
        ins_right_trocar = ['RT - ' + ins for ins in self.ins_names_trocar]
        self.features_to_use = self.ins_names + phases_names + action_names + ins_left_trocar + ins_right_trocar + gen_surg_actphas_names
        self.feature_name_to_num = {}
        for idx, feature_name in enumerate(self.features_to_use):
            self.feature_name_to_num[feature_name] = idx

        if mode == "train":
            self.video_ids = train_ids
        elif mode == "val":
            self.video_ids = val_ids
        elif mode == "test":
            self.video_ids = test_ids
        elif mode == "all":
            self.video_ids = all_video_ids

        self.img_list, self.anno_list = self.get_img_list()


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = torch.load(img_path)
        img_trans = self.transformations[self.mode](img.permute((2, 0, 1)))

        label = self.anno_list[idx]
        id = label['id']
        label_noid = label.drop(['id'])

        label_noid = label_noid[self.features_to_use]
        label_tensor = torch.Tensor(label_noid)
        return img_trans, label_tensor, id


    def get_transformations(self):
        norm_mean = [0.3456, 0.2281, 0.2233]
        norm_std = [0.2528, 0.2135, 0.2104]
        normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
        augmentation_options = {
            'vflip': transforms.RandomVerticalFlip(0.4),
            'hflip': transforms.RandomHorizontalFlip(0.4),
            'colorjitter': transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
            'rot90': transforms.RandomRotation(90,expand=True),
            'brightness': transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.5),
            'contrast': transforms.RandomAutocontrast(p=0.5),
            'randomaffine': transforms.RandomAffine(
                15,
                translate=(0.1, 0.1),
                scale=(0.8,1.5))
        }

        if self.hparams.augmentation_type == "rendezvous":
            aug_list = ['vflip', 'hflip', 'colorjitter', 'rot90']
            aug_prob = 1.0
            resize_height = 256
            resize_width = 448
        elif self.hparams.augmentation_type == "tecno":
            aug_list = ['randomaffine']
            aug_prob  = 0.7
            resize_height = 244
            resize_width = 244
        data_augmentations = [augmentation_options[aug] for aug in aug_list]

        data_transformations = {}
        data_transformations["train"] = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(resize_height, resize_width)),
            transforms.RandomApply([transforms.Compose(data_augmentations)], p=aug_prob),
            transforms.Resize(size=(resize_height, resize_width)),
            transforms.ToTensor(),
            normalize
        ])
        data_transformations["val"] = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(resize_height, resize_width)),
            transforms.ToTensor(),
            normalize
        ])
        data_transformations["test"] = data_transformations["val"]
        data_transformations["all"] = data_transformations["val"]
        return data_transformations


    def get_img_list(self):
        """Returns lists containig image paths and corresponding annoations"""

        # Create directory for pregenerated img list, if not already exists
        os.makedirs(self.pregenerated_list_path, exist_ok=True)

        img_anno_list_path = os.path.join(self.pregenerated_list_path, f"img_anno_list_{self.mode}.pickle")

        # Check if the pickled lists already exist 
        if os.path.isfile(img_anno_list_path):
            # Load the existing lists
            with open(img_anno_list_path, "rb") as fp:
                img_list, anno_list = pickle.load(fp)
        else:
            # Generate lists
            img_list, anno_list = self.generate_imganno_lists()

            # Save the image and annotation lists to a file
            with open(img_anno_list_path, "wb") as fp:
                pickle.dump([img_list, anno_list], fp)

        return img_list, anno_list

    def generate_imganno_lists(self):
        """Generates lists containing the image paths and annotations"""
        # Initialize empty image and annotation lists
        img_list = []
        anno_list = []

        # Iterate through the directories in img_tensor_dir
        for basedir, _, _ in os.walk(self.img_tensor_dir):
            dir_id = basedir.split("/")[-1][1:]

            # Check if the current directory is in video_ids
            if dir_id not in self.video_ids:
                continue

            # Read the corresponding annotation files
            anno_path = os.path.join(self.anno_dir, f"anno_{dir_id}.csv" )
            anno = pd.read_csv(anno_path, index_col=0)  

            print(f"Loading video {dir_id} with {anno.shape[0]} frames")

            # Iterate through the annotation rows
            for _, anno_row in anno.iterrows():
                img_tensor_path = os.path.join(basedir, anno_row.id + '.pt')

                # Check if the image file exists (sometimes the annotations are a bit longer than the video)
                if os.path.isfile(img_tensor_path):
                    assert 'id' in anno_row
                    img_list.append(img_tensor_path)

                    #right trocar
                    right_trocar = torch.tensor(anno_row['Trocar Right'])
                    right_trocar_1h_pd = F.one_hot(right_trocar, num_classes=self.hparams.num_instruments)
                    right_trocar_1h = pd.DataFrame(right_trocar_1h_pd, index= ['RT - ' + ins for ins in self.ins_names_trocar])

                    #left trocar
                    left_trocar = torch.tensor(anno_row['Trocar Left'])
                    left_trocar_1h_pd = F.one_hot(left_trocar, num_classes=self.hparams.num_instruments)
                    left_trocar_1h = pd.DataFrame(left_trocar_1h_pd, index= ['LT - ' + ins for ins in self.ins_names_trocar])

                    anno_row_rtlt = pd.concat([anno_row, left_trocar_1h, right_trocar_1h]).squeeze()

                    anno_list.append(anno_row_rtlt)
                else:
                    print(f"No frame for annotation {anno_row.id}")
        return img_list, anno_list


    def load_split_ids(self):
        """Loads ids for datasplit into file

        Returns:
            lists
        """
        train_path = os.path.join(self.split_dir, "train.txt")
        with open(train_path, 'r') as file:
            train_ids = [line.strip() for line in file]

        val_path = os.path.join(self.split_dir, "val.txt")
        with open(val_path, 'r') as file:
            val_ids = [line.strip() for line in file]

        test_path = os.path.join(self.split_dir, "test.txt")
        with open(test_path, 'r') as file:
            test_ids = [line.strip() for line in file]

        all_ids = train_ids + val_ids + test_ids

        return train_ids, val_ids, test_ids, all_ids


if __name__ == "__main__":
    data_m = VisualDataset()
