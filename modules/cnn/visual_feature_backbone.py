"""Feature extraction for visual feature module adapted from TeCNO: https://github.com/tobiascz/TeCNO"""

import os
from pathlib import Path
import torch
import torchmetrics
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import pickle
from tqdm import tqdm
import wandb
from modules.utils.modelsaving import ModelSaving
from modules.utils.earlystopping import EarlyStopping
#from torch_lr_finder import LRFinder


class FeatureExtraction():
    """Feature extraction class"""
    def __init__(self, hparams, model, dataset):
        """Initializes feature extraction class

        Args:
            hparams: hyperparameters
            model: model to be used for training                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      aining/evaluation
            dataset: Visual dataset
        """
        super(FeatureExtraction, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hparams = hparams
        self.model = model.to(self.device)
        self.dataset = dataset

        self.init_metrics()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.trainloader, self.valloader, self.testloader, self.allloader = self.create_dataloaders()
        self.optimizer = self.configure_optimizers()
        self.early_stopping = EarlyStopping(hparams)
        self.modelsaving = ModelSaving(hparams)

        #learning rate finder
        #self.get_learning_rate()

        # store model
        self.pickle_path = None

        #Logging
        self.writer = SummaryWriter(log_dir=self.hparams.output_path)
        wandb.init(config=self.hparams, project=self.hparams.wandbprojectname, name=self.hparams.name, mode=self.hparams.wandb_mode, dir=self.hparams.output_path)

    '''
    def get_learning_rate(self):
        model = self.model
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
        lr_finder = LRFinder(model, optimizer, criterion, device=self.device)
        lr_finder.range_test(self.trainloader, end_lr=100, num_iter=100)
        lr_finder.plot() # to inspect the loss-learning rate graph
        lr_finder.reset() # to reset the model and optimizer to their initial state
    '''

    def trainval(self):
        """Main training routine"""

        for epoch in tqdm(range(1, self.hparams.max_epochs + 1)):
            self.model.train()
            train_raw_values_list = []
            val_raw_values_list = []
            for train_batch in tqdm(self.trainloader, leave=False):
                train_p_values, train_y_values = self.training_step(train_batch)
                train_raw_values_list.append([train_p_values, train_y_values])
            self.model.eval()
            with torch.no_grad():
                for val_batch in tqdm(self.valloader, leave=False):
                    val_p_values, val_y_values = self.validation_step(val_batch)
                    val_raw_values_list.append([val_p_values, val_y_values])
            stop_early = self.end_of_epoch(train_raw_values_list, val_raw_values_list, epoch)
            if stop_early and (epoch >= self.hparams.min_epochs):
                break
        if self.hparams.extract_features:
            self.extract_features()


    def init_metrics(self):
        """Creates metrics"""
        self.metrics_names = ['acc', 'f1', 'avgprecision', 'auroc', 'recall', 'specificity']
        self.train_metrics_names = ['train_' + metric_name for metric_name in self.metrics_names]
        self.val_metrics_names = ['val_' + metric_name for metric_name in self.metrics_names]
        self.train_metrics_names.append('train_loss')
        self.val_metrics_names.append('val_loss')

        features_to_use = self.dataset.data['train'].features_to_use
        self.train_feature_metrics_names = ['train_' + metric_name + '_' + feature_to_use for metric_name in self.metrics_names for feature_to_use in features_to_use]
        self.val_feature_metrics_names = ['val_' + metric_name + '_' + feature_to_use for metric_name in self.metrics_names for feature_to_use in features_to_use]

        self.acc = torchmetrics.Accuracy(task="binary").to(self.device)
        self.f1 = torchmetrics.F1Score(task="binary").to(self.device)
        self.avgprecision = torchmetrics.AveragePrecision(task="binary").to(self.device)
        self.auroc = torchmetrics.AUROC(task="binary").to(self.device)
        self.recall = torchmetrics.Recall(task="binary").to(self.device)
        self.specificity = torchmetrics.Specificity(task="binary").to(self.device)


    def end_of_epoch(self, train_raw_values_list, val_raw_values_list, epoch):
        """Gets called at the end of epoch
        Calculates and logs metrics
        Saves model if best
        Possibly stops early

        Args:
            train_metrics_dicts: training metrics
            val_metrics_dicts: validation metrics
            epoch: current epoch

        Returns:
            stop_early (bool): whether to stop early
        """
        train_p_values_list = []
        train_y_values_list = []
        val_p_values_list = []
        val_y_values_list = []
        for sublist in train_raw_values_list:
            train_p_values_list.extend(sublist[0])
            train_y_values_list.extend(sublist[1])
        for sublist in val_raw_values_list:
            val_p_values_list.extend(sublist[0])
            val_y_values_list.extend(sublist[1])

        train_metric_dict = self.calc_metrics(torch.stack(train_p_values_list), torch.stack(train_y_values_list), 'train')
        val_metric_dict = self.calc_metrics(torch.stack(val_p_values_list), torch.stack(val_y_values_list), 'val')

        metric_dict = {**train_metric_dict, **val_metric_dict}

        wandb.log(metric_dict)

        self.modelsaving(metric_dict, self.model)
        stop_early = self.early_stopping(metric_dict)
        return stop_early


    def calc_metrics(self, p_values_tensor, y_values_tensor, mode):
        #Calculate metrics
        metric_dict = {}

        #Calculate loss
        metric_dict[f"{mode}_loss"] = self.bce_loss(p_values_tensor, y_values_tensor)
        
        #Sigmoid activation for correct calculation of other metrics
        p_values_tensor = torch.sigmoid(p_values_tensor)

        #Overall metrics
        metric_dict[f"{mode}_acc"] = self.acc(p_values_tensor, y_values_tensor)
        metric_dict[f"{mode}_f1"] = self.f1(p_values_tensor, y_values_tensor)
        metric_dict[f"{mode}_avgprecision"] = self.avgprecision(p_values_tensor, y_values_tensor.long())
        metric_dict[f"{mode}_auroc"] = self.auroc(p_values_tensor, y_values_tensor)
        metric_dict[f"{mode}_recall"] = self.recall(p_values_tensor, y_values_tensor)
        metric_dict[f"{mode}_specificity"] = self.specificity(p_values_tensor, y_values_tensor)

        if mode == "train":
            feature_metrics_names = self.train_feature_metrics_names
        elif mode == "val":
            feature_metrics_names = self.val_feature_metrics_names

        for feature_metric_name in feature_metrics_names:
            single_feature_index = self.dataset.data['train'].feature_name_to_num[feature_metric_name.split('_')[-1]]
            p_single_feature = p_values_tensor[:,single_feature_index]
            y_single_feature = y_values_tensor[:, single_feature_index]
            if (f"{mode}_acc_") in feature_metric_name:
                metric_dict[feature_metric_name] = self.acc(p_single_feature, y_single_feature)
            elif (f"{mode}_f1_") in feature_metric_name:
                metric_dict[feature_metric_name] = self.f1(p_single_feature, y_single_feature) 
            elif (f"{mode}_avgprecision_") in feature_metric_name:
                metric_dict[feature_metric_name] = self.avgprecision(p_single_feature, y_single_feature.long())
            elif (f"{mode}_auroc_") in feature_metric_name:
                metric_dict[feature_metric_name] = self.auroc(p_single_feature, y_single_feature)
            elif (f"{mode}_recall_") in feature_metric_name:
                metric_dict[feature_metric_name] = self.recall(p_single_feature, y_single_feature)
            elif (f"{mode}_specificity_") in feature_metric_name:
                metric_dict[feature_metric_name] = self.specificity(p_single_feature, y_single_feature)
        return metric_dict

    # ---------------------
    # TRAINING
    # ---------------------

    def compute_loss(self, p_features, y_features):
        """Calculates loss

        Args:
            p_features: predicted features
            y_features: true features

        Returns:
            loss: loss value
        """
        loss = self.bce_loss(p_features, y_features)
        return loss

    def training_step(self, batch):
        """Does one training step

        Args:
            batch (list): Contains image tensor, ground truth phases and tools

        Returns:
            metric_dict: dict of training metrics
        """
        x, y_features, id = batch
        x, y_features = x.to(self.device), y_features.to(self.device)

        self.optimizer.zero_grad()
        stem, p_features = self.model.forward(x)
        train_loss = self.compute_loss(p_features, y_features)
        train_loss.backward()
        self.optimizer.step()

        return p_features, y_features

    def validation_step(self, batch):
        """Does one validation step

        Args:
            batch (list): Contains image tensor, ground truth phases and tools

        Returns:
            metric_dict: dict of training metrics
        """
        x, y_features, id = batch
        x, y_features = x.to(self.device), y_features.to(self.device)

        stem, p_features = self.model.forward(x)

        return p_features, y_features


    def save_to_drive(self, vid_index, stems_1vid, preds_1vid):
        """Saves features and predicted phases and tools to drive 

        Args:
            vid_index: index of video
        """
        save_path = Path(self.hparams.data_root) / "stems_and_preds"
        save_path.mkdir(exist_ok=True)
        save_path_vid = save_path / f"video_{vid_index}.pkl"

        with open(save_path_vid, 'wb') as f:
            pickle.dump([
                torch.stack(stems_1vid),
                torch.stack(preds_1vid),
            ], f)


    def extract_features(self):
        """Loops over batches"""
        best_model_path = os.path.join(self.hparams.output_path, "models", "best_model.pt")
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        stems_1vid = []
        preds_1vid = []

        current_video_id = None
        last_frame_id = -1

        with torch.no_grad():
            print("Extracting features:")
            for batch in self.allloader:
                x, y_features, ids = batch
                x, y_features = x.to(self.device), y_features.to(self.device)

                stem, p_features = self.model.forward(x)

                for idx, id in enumerate(ids):
                    video_id = id[:id.find('_')]
                    frame_id = int(id[(id.find('f')+1):])
                    if current_video_id and (video_id != current_video_id):
                        print(f"Saving video {current_video_id}")
                        self.save_to_drive(current_video_id, stems_1vid, preds_1vid)
                        stems_1vid = []
                        preds_1vid = []
                        last_frame_id = -1
                    current_video_id = video_id

                    if (frame_id < last_frame_id):
                        raise Exception("During extraction: Frames seem to be in wrong order")
                    else:
                        last_frame_id = frame_id

                    stems_1vid.append(stem[idx, :])
                    preds_1vid.append(p_features[idx, :])
            # Save last video
            print(f"Saving video {current_video_id}")
            self.save_to_drive(current_video_id, stems_1vid, preds_1vid)

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.hparams.learning_rate)
        return optimizer

    def create_dataloaders(self):
        """Creates dataloaders

        Returns:
            3 dataloaders for training, validation and whole dataset
        """
        trainloader = DataLoader(
            dataset = self.dataset.data["train"],
            batch_size = self.hparams.batch_size,
            shuffle=True,
            num_workers = self.hparams.num_workers,
            pin_memory = True
        )
        valloader = DataLoader(
            dataset = self.dataset.data["val"],
            batch_size = self.hparams.batch_size,
            shuffle=False,
            num_workers = self.hparams.num_workers,
            pin_memory = True
        )
        testloader = DataLoader(
            dataset = self.dataset.data["test"],
            batch_size = self.hparams.batch_size,
            shuffle = False,
            num_workers = self.hparams.num_workers,
            pin_memory = True
        )
        allloader = DataLoader(
            dataset = self.dataset.data["all"],
            batch_size = self.hparams.batch_size,
            shuffle = False,
            num_workers = self.hparams.num_workers,
            pin_memory = True
        )
        return trainloader, valloader, testloader, allloader

    @staticmethod
    def add_module_specific_args(parser):  # pragma: no cover
        """Adds module specific args"""
        module_specific_args = parser.add_argument_group(
            title='module specific args options')
        module_specific_args.add_argument("--learning_rate",
                                      default=0.0005,
                                      type=float)
        module_specific_args.add_argument("--batch_size", default=32, type=int)
        module_specific_args.add_argument("--extract_features", action="store_true")
        module_specific_args.add_argument("--wandb_mode",
                                         default="online",
                                         choices=["online", "offline", "disabled"],
                                         type=str)
        module_specific_args.add_argument("--wandbprojectname",
                                          default='trash',
                                          type=str)
        module_specific_args.add_argument("--do_sweep",
                                          action="store_true")
        #Early stopping
        module_specific_args.add_argument("--early_stopping",
                                         action="store_true")
        module_specific_args.add_argument("--early_stopping_mode",
                                         default = "max",
                                         choices=["min", "max"],
                                         type=str)
        module_specific_args.add_argument("--early_stopping_metric",
                                         default="val_loss",
                                         type=str)
        module_specific_args.add_argument("--early_stopping_patience",
                                         default=5,
                                         type=int)
        module_specific_args.add_argument("--early_stopping_delta",
                                         default=0.0,
                                         type=float)
        #Saving
        module_specific_args.add_argument("--save_model_metric",
                                         default="val_jaccardindex",
                                         type=str)
        module_specific_args.add_argument("--save_model_mode",
                                         default="max",
                                         choices=["min", "max"],
                                         type=str)
        module_specific_args.add_argument("--save_best_model",
                                         action="store_true")
        module_specific_args.add_argument("--testmode",
                                          action="store_true")
        return parser
