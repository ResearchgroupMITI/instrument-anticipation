"""dataset loader for temporal multi-frame model"""

import os
import pickle
import pandas as pd
import torch
from skimage import  measure
from tqdm import tqdm

class AllFeatureDataset():
    def __init__(self, hparams):
        self.name = "AllFeatures"
        self.hparams = hparams
        self.data = self.create_data_sets()


    # create data loaders for all the 3 data splits
    def create_data_sets(self):
        datasets = {}
        for split in ["train", "val", "test"]:
            # create feature dataset for each split
            datasets[split] = AllFeatureDatasetHelper(self.hparams, split)
        return datasets


    @staticmethod
    def add_dataset_specific_args(parser):
        allfeaturedataset_specific_args = parser.add_argument_group(
            title='AllFeatureDataset specific args options'
        )
        allfeaturedataset_specific_args.add_argument("--load_features_from_file",
                                                     action="store_true")
        allfeaturedataset_specific_args.add_argument("--save_features_to_file",
                                                     action="store_true")
        allfeaturedataset_specific_args.add_argument("--cnn_feature_dir",
                                                     default= "stems_and_preds",
                                                     type=str)
        allfeaturedataset_specific_args.add_argument("--segmentation_tensor_dir",
                                                     default="yolo_segment",
                                                     type=str)
        allfeaturedataset_specific_args.add_argument("--yolo_anno_path",
                                                     default="yolo_detect",
                                                     type=str)
        allfeaturedataset_specific_args.add_argument("--feature_save_dir",
                                                     default="../prepro/features_for_stage_2",
                                                     type=str)
        allfeaturedataset_specific_args.add_argument("--num_seg_classes",
                                                     default=34,
                                                     type=int)
        allfeaturedataset_specific_args.add_argument("--ant_delta",
                                                     default=3,
                                                     type=int)
        allfeaturedataset_specific_args.add_argument("--ant_tau",
                                                     default=3,
                                                     type=int)
        allfeaturedataset_specific_args.add_argument("--ant_gamma",
                                                     default=8,
                                                     type=int)
        allfeaturedataset_specific_args.add_argument("--ant_fut",
                                                     default=3,
                                                     type=int)
        allfeaturedataset_specific_args.add_argument("--ant_phase_buffer",
                                                     default=0,
                                                     type=int)
        return parser


# Dataset for training of temporal models using features extracted from CNN (stage 1) - LTC, MSTCN
class AllFeatureDatasetHelper(torch.utils.data.Dataset):
    """Data loader for temporal models"""
    def __init__(self, hparams, mode):
        self.hparams = hparams
        self.split_dir = os.path.join(self.hparams.data_root, 'data_split')

        train_ids, val_ids, test_ids, all_video_ids = self.load_split_ids()

        self.trocar_column_names = ['Trocar Right', 'Trocar Left']

        self.phases_names = ['Phase Trocar Insertion', 'Phase Preparation', 'Phase Calot Triangle Dissection', 'Phase Clipping', 'Phase Dissection',
                'Phase Haemostasis', 'Phase Packaging', 'Phase Retraction', 'Phase Instrument ReInsertion', 'Phase Haemostasis 2', 'Phase Trocar Removal']
 
        if mode == "train":
            self.mode_video_ids = train_ids
        elif mode == "val":
            self.mode_video_ids = val_ids
        elif mode == "test":
            self.mode_video_ids = test_ids

        self.features = []
        self.cnn_predictions = []
        self.tool_gt_all = []
        self.phase_gt_all = []
        self.video_ids = []
        self.mode = mode
        self.data_loaded = False
        self.load_data()


    def __len__(self):
        return len(self.tool_gt_all)


    # get one video features, labels, cnn predictions 
    def __getitem__(self, idx):
        tool_gt = self.tool_gt_all[idx]
        phase_gt = self.phase_gt_all[idx]
        targets = [tool_gt, phase_gt]

        feature_tensor = self.features[idx]

        cnn_predictions = self.cnn_predictions[idx]

        video_id = self.video_ids[idx]

        feature_tensor = torch.cat([feature_tensor, cnn_predictions], dim=1)

        return feature_tensor, targets, cnn_predictions, video_id


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


    def load_data(self):
        """Either generates feature vector from subsystem features or loads feature_vector from file

        Raises:
            Exception: The feature tensor to load can't be found
        """
        feature_save_path = os.path.join(self.hparams.feature_save_dir, self.mode + '_features.pt')
        if self.hparams.load_features_from_file:
            if not os.path.isfile(feature_save_path):
                raise Exception("You set the load_features_from_file flag, but there is no feature file at the given path.")
            features = torch.load(feature_save_path)
            self.features, self.cnn_predictions, self.tool_gt_all, self.phase_gt_all, self.video_ids = features
        else: 
            self.generate_feature_data()
            if self.hparams.save_features_to_file:
                torch.save([
                    self.features, 
                    self.cnn_predictions, 
                    self.tool_gt_all,
                    self.phase_gt_all,
                    self.video_ids
                            ], feature_save_path)
            pass


    # load all video features, labels, cnn predictions
    def generate_feature_data(self):
        """Loads the following features for the model:
            - self.features (list [num_videos]): list of feature tensors of shape [len_sequence, num_features]
            - self.cnn_predictions_tools (list [num_videos]): List of frame-wise tool predictions from cnn of shape [len_sequence, num_tools]
            - self.cnn_predictions_phases (list [num_videos]): List of frame-wise phase predictions from cnn of shape [len_sequence, num_phases]
        Also loads ground truth annotations
        """
        cnn_feature_path = os.path.join(self.hparams.data_root, self.hparams.cnn_feature_dir)
        segmentation_masks_path = os.path.join(self.hparams.data_root, self.hparams.segmentation_tensor_dir)
        yolo_anno_path = os.path.join(self.hparams.data_root, self.hparams.yolo_anno_path)

        print("Combining features")
        for vid_ID in tqdm(self.mode_video_ids):
            self.video_ids.append(vid_ID)

            #get target annotations
            anno_path = os.path.join("anno", "anno_" + vid_ID + ".csv")
            with open(anno_path, 'r') as anno_file:
                tool_gt, phase_gt = self.load_target_annotation_1vid(anno_file)

            #get cnn features
            cnn_features_1vid_path = os.path.join(cnn_feature_path, "video_v" + vid_ID.lstrip("0") + ".pkl")
            with open(cnn_features_1vid_path, 'rb') as f:
                cnn_features, cnn_pred = pickle.load(f)
            cnn_features, cnn_pred = cnn_features.cpu(), cnn_pred.cpu()

            self.cnn_predictions.append(cnn_pred)

            #get interaction features
            segmentation_masks_1vid_dir = os.path.join(segmentation_masks_path, "results", "v" + vid_ID)
            yolo_labels_1vid_dir = os.path.join(yolo_anno_path, "v" + vid_ID, "labels")
            seg_mask_features, yolo_features = self.load_interaction_1vid(segmentation_masks_1vid_dir, yolo_labels_1vid_dir)

            features = torch.cat([cnn_features, cnn_pred, seg_mask_features, yolo_features], dim=1)

            #due to issues with extraction, sometimes there might be a single frame more in the annotions, this gets cut here
            if tool_gt.shape[0] != features.shape[0]:
                tool_gt = tool_gt[:features.shape[0], :]
                phase_gt = phase_gt[:features.shape[0]]

            self.tool_gt_all.append(tool_gt)
            self.phase_gt_all.append(phase_gt)
            self.features.append(features)

        self.data_loaded = True
        return


    def load_interaction_1vid(self, seg_masks_1vid_dir, yolo_labels_dir):
        """Loads segmentation and yolo features

        Args:
            seg_masks_1vid_dir (string): path to directory where segmentation masks of specific video are stored
            yolo_labels_dir (string): path to directory whre yolo labels of specific video are stored

        Returns:
            seg_features_1_vid [len_seq, num_seg_features]: segmentation features
            yolo_featurs_1_vid [len_seq, num_seg_features]: yolo features
        """
        seg_mask_frame_names = os.listdir(seg_masks_1vid_dir)
        seg_mask_frame_names = [name.split('_')[1] for name in seg_mask_frame_names]
        seg_mask_frame_names = sorted(list(set(seg_mask_frame_names)), key=lambda x: int(x[1:]) )
        yolo_labels_frame_names = sorted(os.listdir(yolo_labels_dir))

        seg_features_1vid = []
        yolo_features_1vid = []
        for seg_mask_frame_name in tqdm(seg_mask_frame_names, leave=False):
            seg_features_1frame = self.load_seg_features_1frame(seg_masks_1vid_dir, seg_mask_frame_name)
            seg_features_1vid.append(seg_features_1frame)
            yolo_features_1frame = self.load_yolo_features_1frame(seg_mask_frame_name, yolo_labels_dir, yolo_labels_frame_names)
            yolo_features_1vid.append(yolo_features_1frame)

        seg_features_1vid = torch.stack(seg_features_1vid)
        yolo_features_1vid = torch.stack(yolo_features_1vid)
        return seg_features_1vid, yolo_features_1vid


    def load_seg_features_1frame(self, seg_masks_1vid_dir, seg_mask_frame_name):
        """Loads the segmentation mask for one frame and extracts features from it

        Args:
            seg_masks_1vid_dir (string): path to directory where masks for video are stored
            seg_mask_frame_name (string): name of frame

        Returns:
            torch Tensor: Contains features [number of segmentation classes, 3]
                For each segmentation class three values are stored: 
                    (0) Number of occurences
                    (1) proportion of area belong to specific class (0 if none)
                    (2) y value for center of mass for class (-1 if class not in mask)
                    (3) x value for center of mass for class (-1 if class not in mask)
        """
        video_name = seg_masks_1vid_dir.split("/")[-1]
        mask_name = video_name + '_' + seg_mask_frame_name + '_mask.pt'
        cls_name = video_name + '_' + seg_mask_frame_name + '_cls.pt'

        mask_path = os.path.join(seg_masks_1vid_dir, mask_name)
        cls_path = os.path.join(seg_masks_1vid_dir, cls_name)

        seg_features = torch.zeros(self.hparams.num_seg_classes, 9)
        seg_features[:, 2:] = -1

        with open(mask_path, "rb") as mask_file, open(cls_path, "rb") as cls_file:
            mask = torch.load(mask_file)
            cls = torch.load(cls_file)

        if cls.nelement() == 0:
            seg_features = seg_features.flatten()
            return seg_features

        frame_height, frame_width = mask.shape[1], mask.shape[2]
        frame_area = frame_height * frame_width
        for c, m in zip(cls, mask):
            c = int(c.item())
            seg_features[c, 0] += 1 #no of regions
            if m.max() == 0:
                continue
            region = measure.regionprops(m.cpu().numpy().astype(int))[0]
            region_area = region.area / frame_area
            if region_area > seg_features[c, 1]:
                seg_features[c, 1] = region_area #share of frame occupied by region
                seg_features[c, 2] = region.centroid[0] / frame_height #relative y position
                seg_features[c, 3] = region.centroid[1] / frame_width
                seg_features[c, 4] = region.eccentricity
                seg_features[c, 5] = region.extent
                seg_features[c, 6] = region.orientation
                seg_features[c, 7] = region.perimeter
                seg_features[c, 8] = region.solidity
 
        seg_features = seg_features.flatten()

        return seg_features


    def load_yolo_features_1frame(self, seg_mask_frame_name, yolo_labels_dir, yolo_labels_frame_names):
        """
        Args:
            seg_mask_frame_name (string): name of text file with yolo annotation
            yolo_labels_dir (string): path to directory where text file is located
            yolo_labels_frame_names (list): list of text files in directory

        Returns:
            torch Tensor[39]: Encoding for yolo properties of instruments, label: x center pos, y center pos, width, height
        """
        yolo_label_name = seg_mask_frame_name[:-3] + '.txt'
        yolo_label_name = yolo_labels_dir.split("/")[-2] + '_' + seg_mask_frame_name + '.txt'

        yolo_label_list = []
        if yolo_label_name in yolo_labels_frame_names:
            with open(os.path.join(yolo_labels_dir, yolo_label_name)) as f:
                for line in f:
                    yolo_label = line.split(" ") #class, x, y, w, h, conf
                    yolo_label_list.append(torch.Tensor([float(x) for x in yolo_label[0:5]]))

        yolo_tensor = torch.full((10, 6), 0).float()
        yolo_tensor[:,1:] = -1
        yolo_label_list = sorted(yolo_label_list, key=lambda x: (x[0], x[3]*x[4])) #smaller first

        for label in yolo_label_list: 
            cls_label = int(label[0])
            yolo_tensor[cls_label, 0] += 1
            yolo_tensor[cls_label, 1:5] = label[1:]
            yolo_tensor[cls_label, 5] = label[3] * label[4]

        yolo_tensor = yolo_tensor.flatten()

        return yolo_tensor


    def load_target_annotation_1vid(self, anno_file):
        """Loads data for one video

        Args:
            anno_file: Object containing path to annotation file

        Returns:
            tool_gt: torch Tensor containing shifted ground truth annotations for trocars
            phase_gt: torch Tensor containing shifted ground truth annotations for phases
        """""

        # Read the csv into a pandas DataFrame
        anno_pd = pd.read_csv(anno_file.name, index_col=0)

        # Get tool and phases columns
        tool_pd = anno_pd[self.trocar_column_names]
        phase_pd = anno_pd[self.phases_names]

        #Transform columns
        tool_tensor = torch.Tensor(tool_pd.values)
        phase_tensor = torch.Tensor(phase_pd.values)
        phase_tensor_1D = self.create_1D_phase_tensor(phase_tensor)

        tool_gt = self.generate_anticipation_gt_tool(tool_tensor)
        phase_gt = self.generate_anticipation_gt_phase(tool_tensor, phase_tensor_1D)

        return tool_gt, phase_gt


    def create_1D_phase_tensor(self, phase_tensor: torch.Tensor) -> torch.Tensor:
        """Creates a 1D tensor containing the present phase (signified by numbers) at each
           time step from the one-hot version
           The phases -> number are as folows:
                Phase Trocar Insertion          -> 1
                Phase Preparation               -> 2
                Phase Calot Triangle Dissection -> 3
                Phase Clipping                  -> 4
                Phase Dissection                -> 5
                Phase Haemostasis               -> 6
                Phase Packaging                 -> 7
                Phase Retraction                -> 8
                Phase Instrument ReInsertion    -> 9
                Phase Haemostasis 2             -> 10
                Phase Trocar Removal            -> 11

        Args:
            phase_tensor (torch.Tensor): contains one-hot phases

        Returns:
            torch.Tensor: contains phases in 1D tensor
        """
        #sanity check
        assert torch.all(torch.sum(phase_tensor, dim=1) == 1)
        phase_tensor_1D = torch.argmax(phase_tensor, dim = 1)
        return phase_tensor_1D


    def generate_anticipation_gt_tool(self, presence_gt: torch.Tensor) -> torch.Tensor:
        """generates tool ground truth for anticipation task"""

        # initialize ground truth tensor
        tool_gt = torch.zeros_like(presence_gt)

        # iterates over both columns "Trocar Right" and "Trocar left"
        for j in range(presence_gt.shape[1]):

            # Find positions where the value changes from non-zero to zero
            change_positions = [i for i in range(1, len(presence_gt[:, j])) \
                                if presence_gt[:, j][i - 1].item() != 0 \
                                and presence_gt[:, j][i].item() == 0]

            # Find position where for first instrument 
            first_position = (presence_gt[:, j] != 0).nonzero()[0][0].item()
            change_positions.insert(0, int(first_position)-6)

            # Set the next non-zero value at change positions in the new array
            for pos in change_positions:
                next_non_zero_index = pos + 1
                while next_non_zero_index < len(presence_gt[:, j]) and presence_gt[:, j][next_non_zero_index].item() == 0:
                    next_non_zero_index += 1
                if next_non_zero_index < len(presence_gt[:, j]):
                    range_1 = torch.arange(pos-(self.hparams.ant_delta+self.hparams.ant_tau), pos-(self.hparams.ant_delta-self.hparams.ant_tau)+self.hparams.ant_fut)
                    range_2 = torch.arange(max(0,pos-(self.hparams.ant_delta+self.hparams.ant_tau+self.hparams.ant_gamma)),
                                            pos-(self.hparams.ant_delta+self.hparams.ant_tau))
                    anticipation_values = torch.full((len(range_1),), presence_gt[:, j][next_non_zero_index].item())
                    do_not_move_values = torch.full((len(range_2),), 9.0)

                    tool_gt[:, j][range_2] = do_not_move_values
                    tool_gt[:, j][range_1] = anticipation_values

        return tool_gt


    def generate_anticipation_gt_phase(self, presence_gt: torch.Tensor, phase_info: torch.Tensor):
        """generates phase ground truth for anticipation task"""

        # initialize ground truth tensor
        phase_gt = phase_info.clone()

        # iterates over both columns "Trocar Right" and "Trocar left"
        for j in range(presence_gt.shape[1]):

            # Find positions where the value changes from non-zero to zero
            change_positions = [i for i in range(1, len(presence_gt[:, j])) \
                                if presence_gt[:, j][i - 1].item() != 0 \
                                and presence_gt[:, j][i].item() == 0]

            # Find position where for first instrument 
            first_position = (presence_gt[:, j] != 0).nonzero()[0][0].item()
            change_positions.insert(0, int(first_position)-6)

            # Set the next non-zero value at change positions in the new array
            for pos in change_positions:
                next_non_zero_index = pos + 1
                while next_non_zero_index < len(presence_gt[:, j]) and presence_gt[:, j][next_non_zero_index].item() == 0:
                    next_non_zero_index += 1
                if next_non_zero_index < len(presence_gt[:, j]):
                    if phase_info[pos-(self.hparams.ant_delta+self.hparams.ant_tau+self.hparams.ant_fut)].item() != phase_info[next_non_zero_index].item():
                        ran = torch.arange(pos-(self.hparams.ant_delta+self.hparams.ant_tau), next_non_zero_index)
                        phase_values = torch.full((len(ran),), phase_info[next_non_zero_index].item())

                        phase_gt[ran] = phase_values

        return phase_gt
