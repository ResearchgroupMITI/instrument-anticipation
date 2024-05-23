from typing import Tuple
import numpy as np
#import warnings
#warnings.filterwarnings("error")

def get_trigger(gt: np.ndarray, pred: np.ndarray, m: int, n: int) -> np.ndarray:
    """function to determine the trigger after m out of n correct predictions
    in a sequence

    input:
    gt: ground truth (either left or right trocar)
    pred: prediction (corresponding left or right trocar)
    m: minimum number of correct predictions
    n: considered number of samples

    output:
    trigger: return of the triggered instrument id at the corresponding timepoint,
    same length as gt and pred array
    """

    # initialize queue variables
    num_time_steps = len(gt)
    trigger = []
    last_n_predictions = []

    # iterate over all predicitions
    for i in range(num_time_steps):
        # Add the i-th prediction to the queue
        last_n_predictions.append(pred[i])

        # if the queue length exceeds n, remove the oldest prediction
        if len(last_n_predictions) > n:
            last_n_predictions.pop(0)

        # get type of prediction and number of prediction types 
        # within the last n predictions
        unique_elements, counts = np.unique(last_n_predictions, return_counts=True)
        # check if a number of prediction exceeds m and if this elements is no 0 or 9
        if np.max(counts) >= m and unique_elements[np.argmax(counts)] not in [0, 9]:
            # check if current prediction could be trigger the robotic motion
            if pred[i] not in [0, 9]:
                trigger.append(pred[i])
            else:
                trigger.append(0)
        else:
            trigger.append(0)

    return np.asarray(trigger)

def get_action(gt: np.ndarray, 
               trigger: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray]:
    """function to get the predicted instruments in resting and prediction window
    as well as the positions of the trigger in the respective windows

    input:
    gt: ground truth (either left or right trocar)
    trigger: trigger time point of predicted instrument (either left or right trigger)

    output:
    eval_rw: array of predicted instruments;
    each entry defines the triggered instrument in a resting window;
    array length is equal to the total number of resting windows

    eval_pw: array of predicted instruments; 
    each entry defines the triggered instrument in a prediction window;
    array length is equal to the total number of prediction windows
    
    eval_origin_gt: array with ground truth values;
    array length is equal to the total number of prediction windows

    track_pos_rw: position at which the trigger was released in the resting window

    track_pos_pw: position at which the trigger was released in the prediciton window
    """

    # Initialize variables
    index = np.zeros_like(gt)
    current_count = 0
    numbers_to_drop = [0, 9]
    numbers_to_keep = [9]

    """Index Array"""

    # Iterate through ground truth array
    for i, value in enumerate(gt):
        if gt[i-1] not in [0, 9] and gt[i] in [0, 9]:
            current_count += 1
        index[i] = current_count

    """Predcition Window"""

    # arrays prediction window
    mask_drop = ~np.isin(gt, numbers_to_drop)
    trigger_filter = trigger[mask_drop]
    index_filter = index[mask_drop]
    gt_filter = gt[mask_drop]

    # initialize index and eval arays prediction window
    unique_values_index = np.unique(index_filter)
    eval_pw = np.zeros(len(unique_values_index), dtype=int)
    eval_origin_gt = np.zeros(len(unique_values_index), dtype=int)
    track_pos_pw = np.zeros(len(unique_values_index), dtype=int)

    # Iterate through unique values in index_filter
    for i, value in enumerate(unique_values_index):
        # Check if there is a prediction in prediction window
        mask_index_filter = (index_filter == value)

        eval_t = trigger_filter[mask_index_filter]
        eval_gt = gt_filter[mask_index_filter]

        # find the indices of the non-zero elements in trigger
        non_zero_indices_t= np.nonzero(eval_t)[0]

        if non_zero_indices_t.size > 0:
            eval_pw[i] = eval_t[non_zero_indices_t[0]]
            track_pos_pw[i] = non_zero_indices_t[0]

        eval_origin_gt[i] = eval_gt[0]

    """Resting Window"""

    # arrays resting window
    mask_keep = np.isin(gt, numbers_to_keep)
    trigger_filter_rw = trigger[mask_keep]
    index_filter_rw = index[mask_keep]

    # initialize index and eval arrays resting window
    unique_values_index_rw = np.unique(index_filter_rw)
    eval_rw = np.zeros(len(unique_values_index), dtype=int)
    track_pos_rw = np.zeros(len(unique_values_index), dtype=int)

    # Iterate through unique values in index_filter
    for i, value in enumerate(unique_values_index_rw):
        # Check if there is a prediction in resting window
        mask_index_filter_rw = (index_filter_rw == value)
        eval_t_rw = trigger_filter_rw[mask_index_filter_rw]

        # find the indices of the non-zero elements in trigger
        non_zero_indices_t_rw= np.nonzero(eval_t_rw)[0]

        if non_zero_indices_t_rw.size > 0:
            eval_rw[i] = eval_t_rw[non_zero_indices_t_rw[0]]
            track_pos_rw[i] = non_zero_indices_t_rw[0]

    return eval_rw, eval_pw, eval_origin_gt

def macro_averaged_metric(rw: np.ndarray, pw:np.ndarray, gt: np.ndarray, troc_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """function to calculate metric scores

    input:
    rw: array of triggered instruments in resting windows
    pw: array of triggered instruments in predcition windows
    gt: array with ground truth of respective windows

    output:
    met_mat: matrix with metric scores for each instrument
    met_max_avgerage: vector with averaged scores over all instruments
    (see metric slide for more information)
    """
    # initialize matrix where metric scores will be stored
    met_mat = np.zeros([8,16])

    # iterate over each instrument
    for j in range(1,9):
        idx = 0
        # initialize metric vectors
        tp_vec = np.zeros(len(gt), dtype=int)
        tp_s_vec = np.zeros(len(gt), dtype=int)
        fn_vec = np.zeros(len(gt), dtype=int)
        fn_s_vec = np.zeros(len(gt), dtype=int)
        fp_vec = np.zeros(len(gt), dtype=int)
        fp_s_vec = np.zeros(len(gt), dtype=int)
        tn_vec = np.zeros(len(gt), dtype=int)
        tn_s_vec = np.zeros(len(gt), dtype=int)
        numb = 0
        # iterate over each entry in gt
        for i, value in enumerate(gt):
            if j == value:
                numb += 1
                if rw[i] == 0 and pw[i] == j:
                    tp_vec[idx] = 1
                elif rw[i] == j:
                    tp_s_vec[idx] = 1
                elif rw[i] == 0 and pw[i] != j:
                    fn_vec[idx] = 1
                elif rw[i] not in [0, j]:
                    fn_s_vec[idx] = 1
            if j != value:
                if rw[i] == 0 and pw[i] == j:
                    fp_vec[idx] = 1
                elif rw[i] == j:
                    fp_s_vec[idx] = 1
                elif rw[i] == 0 and pw[i] != j:
                    tn_vec[idx] = 1
                elif rw[i] not in [0, j]:
                    tn_s_vec[idx] = 1
            idx += 1
            if idx == len(gt):
                met_mat[j-1, 0] = j
                # true positive
                met_mat[j-1, 1] = np.sum(tp_vec)/len(tp_vec)
                # true positive tilde
                met_mat[j-1, 2] = np.sum(tp_s_vec)/len(tp_s_vec)
                # false positive
                met_mat[j-1, 3] = np.sum(fp_vec)/len(fp_vec)
                # false positive tilde
                met_mat[j-1, 4] = np.sum(fp_s_vec)/len(fp_s_vec)
                # false negative
                met_mat[j-1, 5] = np.sum(fn_vec)/len(fn_vec)
                # false negative tilde
                met_mat[j-1, 6] = np.sum(fn_s_vec)/len(fn_s_vec)
                # true negative
                met_mat[j-1, 7] = np.sum(tn_vec)/len(tn_vec)
                # true negative tilde
                met_mat[j-1, 8] = np.sum(tn_s_vec)/len(tn_s_vec)

                # precision
                if np.sum(tp_vec+tp_s_vec+fp_vec+fp_s_vec) != 0:
                    met_mat[j-1, 9] = np.divide(np.sum(tp_vec+tp_s_vec), 
                                                np.sum(tp_vec+tp_s_vec+fp_vec+fp_s_vec))
                else:
                    met_mat[j-1, 9] = 0

                # recall
                if np.sum(tp_vec+tp_s_vec+fn_vec+fn_s_vec) != 0:
                    met_mat[j-1, 10] = np.divide(np.sum(tp_vec+tp_s_vec), 
                                                np.sum(tp_vec+tp_s_vec+fn_vec+fn_s_vec))
                else:
                    met_mat[j-1, 10] = 0

                # specifity 
                if np.sum(tn_vec+tn_s_vec+fp_vec+fp_s_vec) != 0:
                    met_mat[j-1, 11] = np.divide(np.sum(tn_vec+tn_s_vec), 
                                                np.sum(tn_vec+tn_s_vec+fp_vec+fp_s_vec))
                else:
                    met_mat[j-1, 11] = 0

                # negative predictive value
                if np.sum(tn_vec+tn_s_vec+fn_vec+fn_s_vec) != 0:
                    met_mat[j-1, 12] = np.divide(np.sum(tn_vec+tn_s_vec), 
                                                np.sum(tn_vec+tn_s_vec+fn_vec+fn_s_vec))
                else:
                    met_mat[j-1, 12] = 0

                # accuracy
                if np.sum(tp_vec+tp_s_vec+fp_vec+fp_s_vec+fn_vec+fn_s_vec+tn_vec+tn_s_vec) != 0:
                    met_mat[j-1, 13] = np.divide(np.sum(tp_vec+tp_s_vec+tn_vec+tn_s_vec), 
                                                np.sum(tp_vec+tp_s_vec+fp_vec+fp_s_vec+fn_vec+fn_s_vec+tn_vec+tn_s_vec))
                else:
                    met_mat[j-1, 13] = 0

                # f1 score
                if (met_mat[j-1, 9]+met_mat[j-1, 10]) != 0:
                    met_mat[j-1, 14] = np.divide(2*met_mat[j-1, 9]*met_mat[j-1, 10],
                                                met_mat[j-1, 9]+met_mat[j-1, 10])
                else:
                    met_mat[j-1, 14] = 0
                met_mat[j-1, 15] = numb

    # replace nan entries
    met_mat = np.nan_to_num(met_mat)

    # initialize matrix where average metric scores will be stored
    met_mat_average = np.zeros([1, met_mat.shape[1]])

    if troc_name == 'right':
        indices_with_ins = [1, 2, 3, 4, 5, 6, 7]
    elif troc_name == 'left':
        indices_with_ins = [0, 4]
    else:
        raise ValueError('troc_name has to be either "right" or "left".')

    met_mat_ins = met_mat[indices_with_ins, :]

    # iterate over each column in met_mat
    for d in range(met_mat_ins.shape[1]):
        # take the mean of each column
        met_mat_average[0, d] = np.mean(met_mat_ins[:, d])

    return met_mat[:,9:], met_mat_average[:,9:], len(gt)


def weighted_averaged_metric(matrixR, sumR, matrixL, sumL):

    # precision
    prec = (np.dot(matrixR[:, 0], matrixR[:, 6]) + np.dot(matrixL[:, 0], matrixL[:, 6]))/(sumR+sumL)
    # recall
    rec = (np.dot(matrixR[:, 1], matrixR[:, 6]) + np.dot(matrixL[:, 1], matrixL[:, 6]))/(sumR+sumL)
    # accuracy
    acc = (np.dot(matrixR[:, 4], matrixR[:, 6]) + np.dot(matrixL[:, 4], matrixL[:, 6]))/(sumR+sumL)
    # f1-score
    f1 = (np.dot(matrixR[:, 5], matrixR[:, 6]) + np.dot(matrixL[:, 5], matrixL[:, 6]))/(sumR+sumL)

    return [prec, rec, acc, f1]


def calculate_metrics(gt_left, pred_left, gt_right, pred_right, mode):

    ins_names = ['Grasper', 'Biopsy forceps', 'Clipper', 'Scissors', 'Irrigator', 'Retrieval bag', 'Drain', 'SC-tube']
    metric_names = ['Prec', 'Rec', 'Spec', 'NegPredVal', 'Acc', 'F1']
    weighted_metric_names = ['Prec', 'Rec', 'Acc', 'F1']

    #transform everything to numpy
    gt_left = gt_left.cpu().numpy().squeeze()
    pred_left_raw = pred_left.cpu().numpy()
    gt_right = gt_right.cpu().numpy().squeeze()
    pred_right_raw = pred_right.cpu().numpy()

    # Select outputs of last network layer
    pred_left_probs = pred_left_raw[-1, 0, :, :]
    pred_right_probs = pred_right_raw[-1, 0, :, :]

    # Select index of maximum value
    pred_left = np.argmax(pred_left_probs, axis=0)
    pred_right = np.argmax(pred_right_probs, axis=0)

    # 2 von 3, 1 von 1, 3 von 4, 3 von 5, 4 von 5
    mn_names = ["2von3", "1von1", "3von4", "3von5", "4von5"]
    mn_list = [[2, 3], [1, 1], [3, 4], [3, 5], [4, 5]]
    metric_dict = {}

    for mn_name, mn in zip(mn_names, mn_list):
        # get trigger for left trocar
        t_left = get_trigger(gt=gt_left, pred=pred_left, m=mn[0], n=mn[1])

        # get trigger for right trocar
        t_right = get_trigger(gt=gt_right, pred=pred_right, m=mn[0], n=mn[1])

        # get action values for left trocar
        rw_act_ol, pw_act_ol, gt_act_l = get_action(gt=gt_left, trigger=t_left)

        # get action values for right trocar
        rw_act_or, pw_act_or, gt_act_r = get_action(gt=gt_right, trigger=t_right)

        # calculate metric scores for left trocar
        met_mat_l, met_mat_l_avg, lenL = macro_averaged_metric(rw=rw_act_ol, pw=pw_act_ol, gt=gt_act_l, troc_name="left")

        for ins_idx, ins_name in enumerate(ins_names):
            for metric_idx, metric_name in enumerate(metric_names):
                metric_dict[f"{mode}_LeftT_{metric_name}_{ins_name}_{mn_name}"] = met_mat_l[ins_idx, metric_idx]

        for metric_idx, metric_name in enumerate(metric_names):
                metric_dict[f"{mode}_LeftT_{metric_name}_avg_{mn_name}"] = met_mat_l_avg[0, metric_idx]

        # calculate metric scores for right trocar
        met_mat_r, met_mat_r_avg, lenR = macro_averaged_metric(rw=rw_act_or, pw=pw_act_or, gt=gt_act_r, troc_name="right")

        for ins_idx, ins_name in enumerate(ins_names):
            for metric_idx, metric_name in enumerate(metric_names):
                metric_dict[f"{mode}_RightT_{metric_name}_{ins_name}_{mn_name}"] = met_mat_r[ins_idx, metric_idx]

        for metric_idx, metric_name in enumerate(metric_names):
                metric_dict[f"{mode}_RightT_{metric_name}_avg_{mn_name}"] = met_mat_r_avg[0, metric_idx]

        # calculate weighted metric scores
        weighted_m = weighted_averaged_metric(met_mat_r, lenR, met_mat_l, lenL)

        for metric_idx, metric_name in enumerate(weighted_metric_names):
                metric_dict[f"{mode}_weighted_{metric_name}_{mn_name}"] = weighted_m[metric_idx]

    return metric_dict
