import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLoss():
    def __init__(self, hparams, device):
        self.hparams = hparams
        self.device = device
        self.focal_alpha = 1.0
        self.focal_gamma = self.hparams.focal_gamma
        self.nll_loss_func = nn.NLLLoss(weight=None, reduction="none")
        self.phase_loss = nn.CrossEntropyLoss()

    def compute_loss(self, preds, targets):
        """Computes loss

        Args:
            output_reg (torch Tensor): regression output of network [1, len_seq, num_output_features]
            output_cls (torch Tensor): classification output of network [1, len_seq, 3, num_output_features]
            targets (list): contains regression and classification targets

        Returns:
            torch Tensor: Loss value 
        """
        output_lt, output_rt, output_phase = preds
        tool_gt, phase_gt = targets    
        rt_gt = tool_gt[:, :, 0]
        lt_gt = tool_gt[:, :, 1]                                            

        stages = output_lt.shape[0]
        loss = 0

        for j in range(stages):
            stage_loss = 0
            rt_loss = self.compute_loss_one_modality(output_rt[j,:,:,:], rt_gt.long())
            lt_loss = self.compute_loss_one_modality(output_lt[j,:,:,:], lt_gt.long())
            phase_loss = self.compute_loss_phase(output_phase[j,:,:,:], phase_gt.long())

            stage_loss = self.hparams.rt_loss_scale * rt_loss + self.hparams.lt_loss_scale * lt_loss +  self.hparams.phase_loss_scale * phase_loss

            loss += stage_loss
        loss = loss / (stages * 1.0)
        assert ~torch.isinf(loss), "Loss is infinite"
        assert ~torch.isnan(loss), "Loss is NaN"
        return loss

    # First, we need to find the sequences that we use for start and end
    def find_sequences(self, matches: list) -> list:
        #Find start indices
        starts = torch.cat((matches[0:1], matches[1:][torch.diff(matches) > 1]))
        #Find end indices
        ends = torch.cat((matches[:-1][torch.diff(matches) > 1], matches[-1:]))
        #Put together sequences
        sequences = list(zip(starts.tolist(), ends.tolist()))
        return sequences

    def calc_scale_part(self, tensor: torch.Tensor, value: int) -> float:
        #Find all tensor elements with this value
        matches = (tensor == value).nonzero(as_tuple=True)[0]

        # Get continuous sequences of value
        sequences = self.find_sequences(matches)

        scale_part = torch.zeros_like(tensor, dtype=torch.float)
        seq_idx = 0

        # Loop over matches
        for t in matches:
            # Find associated sequence
            while t > sequences[seq_idx][1]:
                seq_idx += 1
            associated_sequence = sequences[seq_idx]

            #Get indices
            start_idx = associated_sequence[0]
            end_idx = associated_sequence[1]

            #Calculate scale (for either act or rest)
            if start_idx != end_idx: # Avoid 0/0 calculation for sequence of length 1
                assert 0 <= self.hparams.loss_scale_start_value <= 1.0, "loss_scale_start value can not be negative or larger than one"
                loss_scale_growth_factor = 1.0 - self.hparams.loss_scale_start_value
                scale_part_t = self.hparams.loss_scale_start_value + ((((start_idx - t)/(end_idx-start_idx))**2) * loss_scale_growth_factor)
            else:
                scale_part_t = 1.0
            scale_part[t] = scale_part_t
        return scale_part * self.hparams.lambda_pred

    def calc_scale_part_rest(self, tensor: torch.Tensor, value: int) -> float:
        #Find all tensor elements with this value
        matches = (tensor == value).nonzero(as_tuple=True)[0]

        # Get continuous sequences of value
        sequences = self.find_sequences(matches)

        scale_part = torch.zeros_like(tensor, dtype=torch.float)
        seq_idx = 0

        # Loop over matches
        for t in matches:
            # Find associated sequence
            while t > sequences[seq_idx][1]:
                seq_idx += 1
            associated_sequence = sequences[seq_idx]

            #Get indices
            start_idx = associated_sequence[0]
            end_idx = associated_sequence[1]

            #Calculate scale (for either act or rest)
            if start_idx != end_idx: # Avoid 0/0 calculation for sequence of length 1
                assert 0 <= self.hparams.loss_scale_start_value_rest <= 1.0, "loss_scale_start value can not be negative or larger than one"
                loss_scale_growth_factor = 1.0 - self.hparams.loss_scale_start_value_rest
                scale_part_t = self.hparams.loss_scale_start_value_rest + ((((start_idx - t)/(end_idx-start_idx))**2) * loss_scale_growth_factor)
            else:
                scale_part_t = 1.0
            scale_part[t] = scale_part_t
        return scale_part * self.hparams.lambda_rest

    def calc_scale(self, tensor: torch.Tensor):
        scale_act = sum([self.calc_scale_part(tensor, value) for value in list(range(1,9))])
        scale_rest = self.calc_scale_part_rest(tensor, 9)
        scale_ign = (tensor == 0) * self.hparams.loss_ign_scale_value

        scale = scale_act + scale_rest + scale_ign

        return scale

    def calc_focal_loss_nll(self, inputs, targets):
        # compute weighted cross entropy term: -alpha * log(pt)
        if self.hparams.use_human_knowledge:
            log_p = torch.log(inputs)
        else:
            log_p = torch.log(F.softmax(inputs, dim=-1))

        nll = self.nll_loss_func(log_p, targets.long())

        # get true class column from each row
        all_rows = torch.arange(len(inputs))
        log_pt = log_p[all_rows, targets.long()]

        #create gamma tensor, which has 2 for val != 0
        gamma_tensor = torch.ones(targets.shape).to(self.device)
        gamma_tensor[targets != 0] = self.focal_gamma

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**gamma_tensor

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * nll
        return loss

    def compute_loss_phase(self, inputs, phase_gt):
        loss = self.phase_loss(inputs, phase_gt)
        return loss

    def compute_loss_one_modality(self, inputs, targets):
        #Change tensors to fit
        inputs = inputs.transpose(1, 2).squeeze(0)
        targets = targets.squeeze(0)
        #Calculate loss and scale
        loss = self.calc_focal_loss_nll(inputs, targets)
        scale = self.calc_scale(targets)

        #Multiply Loss and Scale
        total_loss = loss * scale
        total_loss = total_loss.mean()
        return total_loss
