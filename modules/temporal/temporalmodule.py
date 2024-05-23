import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from modules.utils.schedulers import CosineWarmupScheduler
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import wandb
from tqdm import tqdm
from modules.utils.earlystopping import EarlyStopping
from modules.utils.modelsaving import ModelSaving
from modules.utils.temporal_loss import TemporalLoss
from modules.utils.temporal_metrics import calculate_metrics

class TemporalModule():
    def __init__(self, hparams, model, dataset):
        super(TemporalModule, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hparams = hparams
        self.dataset = dataset
        self.model = model.to(self.device)
        if self.hparams.testmode is True:
            self.model.load_state_dict(torch.load("logs/best_model.pt"))

        self.trainloader, self.valloader, self.testloader = self.create_dataloaders()
        self.temporal_loss = TemporalLoss(hparams, self.device)

        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        self.early_stopping = EarlyStopping(hparams)
        self.modelsaving = ModelSaving(hparams) 

        self.comp_tensor_left, self.comp_tensor_right = self.create_hk_dict()

        #Metrics
        self.ins_names = ['Grasper', 'Biopsy forceps', 'Clipper', 'Scissors', 'Irrigator',
                   'Retrieval bag', 'Drain', 'SC-tube']
        self.train_stat_list = []
        self.val_stat_list = []
        self.test_stat_list = []


    def create_hk_dict(self):
        """Loads human knowledge dict"""
        comp_matrix_left = pd.read_csv("utils/Comp_left.csv", sep=";", index_col=0)
        comp_matrix_right = pd.read_csv("utils/Comp_right.csv", sep=";", index_col=0)
        comp_tensor_left = torch.Tensor(comp_matrix_left.values).to(self.device)
        comp_tensor_right = torch.Tensor(comp_matrix_right.values).to(self.device)

        return comp_tensor_left, comp_tensor_right


    def trainval(self):
        """Main loop for training and evaluation"""
        # Loop over epochs
        for epoch in tqdm(range(1, self.hparams.max_epochs +1)):
            # Set model to training mode and loop over training data loader
            self.model.train()
            for train_batch in tqdm(self.trainloader, leave=False):
                preds, targets, loss = self.training_step(train_batch)
                self.update_metrics(preds, targets, loss, "train")

            # Set model to evaluation mode and loop over validation data loader
            self.model.eval()
            with torch.no_grad():
                for val_batch in tqdm(self.valloader, leave=False):
                    preds, targets, loss = self.validation_step(val_batch)
                    self.update_metrics(preds, targets, loss, "val")

            #Executes after one epoch and stops early, if necessary
            stop_early = self.end_of_epoch(epoch)
            if stop_early and (epoch >= self.hparams.min_epochs):
                break


    def test(self):
        self.model.eval()
        with torch.no_grad():
            for val_batch in tqdm(self.valloader, leave=False):
                preds, targets, loss = self.validation_step(val_batch)
                self.update_metrics(preds, targets, loss, "val")

            for test_batch in tqdm(self.testloader, leave=False):
                preds, targets, loss = self.validation_step(test_batch)
                self.update_metrics(preds, targets, loss, "test")
                if self.hparams.save_test_tensors:
                    os.makedirs(os.path.join(self.hparams.output_path, "test_tensors"), exist_ok=True)
                    save_path_preds = os.path.join(self.hparams.output_path, "test_tensors", f"preds_{test_batch[3][0]}.pt")
                    save_path_targets = os.path.join(self.hparams.output_path, "test_tensors", f"targets_{test_batch[3][0]}.pt")
                    torch.save(preds, save_path_preds)
                    torch.save(targets, save_path_targets)

        _ = self.end_of_epoch(1)


    def forward(self, x):
        """Forward pass through model

        Args:
            x (torch Tensor): made up of input features [1, len_seq, num_input_features]

        Returns:
            output_reg : regression predictions [1, len_seq, num_output_features]
            output_cls : classification predictions [1, len_seq, 3, num_output_features]
        """
        output = self.model(x)
        output_lt = output[:, :, :10, :]
        output_rt = output[:, :, 10:20, :]
        output_phase = output[:, :, 20:, :]

        if self.hparams.use_human_knowledge:
            output_lt, output_rt = self.integrate_human_knowledge(output_lt, output_rt, output_phase)

        return [output_lt, output_rt, output_phase]


    def integrate_hk_1troc(self, output_1t, output_phase, comp_tensor):
        """Integrates human knowledge for one trocar"""
        #Adds idle and resting classes for human knowledge matrix
        comp_tensor_09 = torch.ones(comp_tensor.shape[0]+2, comp_tensor.shape[1])
        comp_tensor_09[1:-1,:] = comp_tensor

        #Calculate the adjusted output probabilities from the phase information
        phase_probs = torch.softmax(output_phase, dim=-2)
        phasecomp = torch.matmul(comp_tensor_09.to(self.device), phase_probs)
        output_probs = torch.softmax(output_1t, dim=-2)
        hk_adjusted_output = torch.mul(phasecomp, output_probs)

        #Replace zeros with very small value to avoid numerical instabilities
        zero_mask = hk_adjusted_output == 0
        hk_adjusted_output[zero_mask] = 1e-37

        #Normalize output
        sums = torch.sum(hk_adjusted_output, dim=-2)
        sums_broadcasted = sums.unsqueeze(2).repeat(1, 1, 10, 1)
        hk_adjusted_output_norm = hk_adjusted_output / sums_broadcasted

        return hk_adjusted_output_norm


    def integrate_human_knowledge(self, output_lt, output_rt, output_phase):
        """Wrapper to integrate human knowledge for both trocars"""
        output_lt_hk = self.integrate_hk_1troc(output_lt, output_phase, self.comp_tensor_left)
        output_rt_hk = self.integrate_hk_1troc(output_rt, output_phase, self.comp_tensor_right)

        return output_lt_hk, output_rt_hk


    def training_step(self, batch):
        """Executes 1 forward pass

        Args:
            batch (list): Contains input data   

        Returns:
            reg_pred (torch Tensor): regression prediction for each timestep
            cls_pred (torch Tensor): classification prediction for each timestep
        """
        # Extracts data from loader and sends it to device
        input, targets, cnn_preds, video_id = batch
        input, targets, cnn_preds = input.to(self.device), [t.to(self.device) for t in targets], [c.to(self.device) for c in cnn_preds]

        # Forward pass and optimization
        self.optimizer.zero_grad()
        preds = self.forward(input)
        train_loss = self.temporal_loss.compute_loss(preds, targets)


        train_loss.backward()
        self.optimizer.step()

        return preds, targets, train_loss

 
    def validation_step(self, batch):
        """Executes 1 validation step

        Args:
            batch (list): Contains input data

        Returns:
            _type_: _description_
        """
        input, targets, cnn_preds, video_idx = batch
        input, targets, cnn_preds = input.to(self.device), [t.to(self.device) for t in targets], [c.to(self.device) for c in cnn_preds]
        preds = self.forward(input)
        val_loss = self.temporal_loss.compute_loss(preds, targets)
        return preds, targets, val_loss
    

    def configure_optimizers(self):
        """Creates optimizer and scheduler

        Returns:
            optimizer, scheduler
        """
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.hparams.learning_rate,
                               weight_decay=self.hparams.weight_decay)
        lr_scheduler =  CosineWarmupScheduler(optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_epochs)
        return optimizer, lr_scheduler


    def end_of_training(self, metrics_dicts):
        for mode in ["train", "val"]:
            goalmetric_pred_all = torch.cat([x[f"{mode}_goalmetric_pred"] for x in metrics_dicts])
            goalmetric_target_all = torch.cat([x[f"{mode}_goalmetric_target"] for x in metrics_dicts])
            wandb.log({f"{mode}_conf_mat" : wandb.plot.confusion_matrix(y_true=goalmetric_target_all.numpy(), preds=goalmetric_pred_all.numpy(), class_names=self.tool_names)})
            pass


    def update_metrics(self, preds, targets, loss, mode):
        if mode == "train":
            self.train_stat_list.append([preds, targets, loss])
        elif mode in "val":
            self.val_stat_list.append([preds, targets, loss])
        elif mode in "test":
            self.test_stat_list.append([preds, targets, loss])
        return


    def calc_special_metrics(self, mode):
        if mode == "train":
            raw_values_list = self.train_stat_list
        elif mode == "val":
            raw_values_list = self.val_stat_list
        elif mode == "test":
            raw_values_list = self.test_stat_list

        metrics_list = []
        loss_list = []

        for singlevid in raw_values_list:
            preds = singlevid[0]
            targets = singlevid[1]
            loss = singlevid[2]

            preds_lt = preds[0].detach()
            preds_rt = preds[1].detach()

            targets_lt = targets[0][:,:,1]
            targets_rt = targets[0][:,:,0]

            metrics = calculate_metrics(targets_lt, preds_lt, targets_rt, preds_rt, mode)
            metrics_list.append(metrics)
            loss_list.append(loss.detach().cpu().numpy())

        loss_mean = np.mean(loss_list)

        metrics_avg_dict = {}

        metrics_names = metrics_list[0].items()
        for key, _ in metrics_names:
            metrics_avg_list = []
            for metrics in metrics_list:
                metrics_avg_list.append(metrics[key])
            metrics_avg = np.mean(metrics_avg_list)
            metrics_avg_dict[key] = metrics_avg

        metrics_avg_dict[f"{mode}_loss"] = loss_mean
        if mode == "train":
            metrics_avg_dict["lr"] = self.optimizer.param_groups[0]['lr']

        return metrics_avg_dict


    def calc_all_metrics(self, mode):
        special_metrics_dict = self.calc_special_metrics(mode)
        return special_metrics_dict


    def end_of_epoch(self, epoch):
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

        if not self.hparams.testmode:
            train_metric_dict = self.calc_all_metrics('train')
        else:
            test_metric_list = self.calc_all_metrics('test')
        val_metric_dict = self.calc_all_metrics('val')

        if not self.hparams.testmode: 
            metric_dict = {**train_metric_dict, **val_metric_dict}
        else:
            metric_dict = {**val_metric_dict, **test_metric_list}

        wandb.log(metric_dict)

        self.train_stat_list = []
        self.val_stat_list = []
        self.test_stat_list = []

        self.lr_scheduler.step()

        self.modelsaving(metric_dict, self.model)
        stop_early = self.early_stopping(metric_dict)
        return stop_early


    def create_dataloaders(self):
        trainloader = DataLoader(
            dataset=self.dataset.data["train"],
            batch_size=1,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
        valloader = DataLoader(
            dataset=self.dataset.data["val"],
            batch_size = 1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
        testloader = DataLoader(
            dataset=self.dataset.data["test"],
            batch_size = 1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
        return trainloader, valloader, testloader

    @staticmethod
    def add_module_specific_args(parser):  # pragma: no cover
        temporalmodule_specific_args = parser.add_argument_group(
            title='Temporal module specific args options'
        )
        temporalmodule_specific_args.add_argument("--learning_rate",
                                                             default=0.0005,
                                                             type=float)
        temporalmodule_specific_args.add_argument("--warmup",
                                                             default=50,
                                                             type=int)
        temporalmodule_specific_args.add_argument("--wandb_mode",
                                                            default="online",
                                                            choices=["online", "offline", "disabled"],
                                                            type=str)
        temporalmodule_specific_args.add_argument("--wandbprojectname",
                                                             default="default",
                                                             type=str)
        temporalmodule_specific_args.add_argument("--weight_decay",
                                                             default=0.0,
                                                             type=float)
        temporalmodule_specific_args.add_argument("--use_human_knowledge",
                                                             action="store_true")
        temporalmodule_specific_args.add_argument("--do_sweep",
                                                             action="store_true")
        temporalmodule_specific_args.add_argument("--sweep_config_path",
                                                             type=str)
        temporalmodule_specific_args.add_argument("--sweep_run_count",
                                                             default=100,
                                                             type=int)
        temporalmodule_specific_args.add_argument("--testmode",
                                                             action="store_true")
        temporalmodule_specific_args.add_argument("--save_test_tensors",
                                                             action="store_true")
        
        #Early stopping stuff
        temporalmodule_specific_args.add_argument("--early_stopping",
                                                            action="store_true")
        temporalmodule_specific_args.add_argument("--early_stopping_mode",
                                                            default = "min",
                                                            choices=["min", "max"],
                                                            type=str)
        temporalmodule_specific_args.add_argument("--early_stopping_metric",
                                                            default="val_loss",
                                                            type=str)
        temporalmodule_specific_args.add_argument("--early_stopping_patience",
                                                            default=5,
                                                            type=int)
        temporalmodule_specific_args.add_argument("--early_stopping_delta",
                                                            default=0.0,
                                                            type=float)
        
        #Saving
        temporalmodule_specific_args.add_argument("--save_model_metric",
                                                            default="val_loss",
                                                            type=str)
        temporalmodule_specific_args.add_argument("--save_model_mode",
                                                            default="min",
                                                            choices=["min", "max"],
                                                            type=str)
        temporalmodule_specific_args.add_argument("--save_best_model",
                                                            action="store_true")
        temporalmodule_specific_args.add_argument("--save_last_model",
                                                            action="store_true")
        
        #Focal loss
        temporalmodule_specific_args.add_argument("--focal_gamma",
                                                             default = 2.0,
                                                             type=float)
        temporalmodule_specific_args.add_argument("--rt_loss_scale",
                                                             default = 1.0,
                                                             type=float)
        temporalmodule_specific_args.add_argument("--lt_loss_scale",
                                                             default = 1.0,
                                                             type=float)
        temporalmodule_specific_args.add_argument("--phase_loss_scale",
                                                             default = 0.2,
                                                             type=float)
        temporalmodule_specific_args.add_argument("--loss_scale_start_value",
                                                             default=0.2,
                                                             type=float)
        temporalmodule_specific_args.add_argument("--loss_scale_start_value_rest",
                                                             default=0.8,
                                                             type=float)
        temporalmodule_specific_args.add_argument("--loss_ign_scale_value",
                                                             default=0.05,
                                                             type=float)
        temporalmodule_specific_args.add_argument("--lambda_pred",
                                                             default=0.2,
                                                             type=float)
        temporalmodule_specific_args.add_argument("--lambda_rest",
                                                             default=2.0,
                                                             type=float)
        return parser
