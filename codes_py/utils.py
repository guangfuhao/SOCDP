from configs import *
import torch
import numpy as np
from model_train import *


def get_activation(datasets, train_idx, epoch, model_name='', class_num=10, shuffle=False, lambda_sparse=0.01,
                   model_config=[], out_dir=''):
    middle_size = model_config[2]
    test_net = initialize_model(model_name, class_num, model_config, lambda_sparse)
    filename = f"model_{train_idx}_epoch_{epoch}.pt"
    path = get_path(out_dir + "model_weights", middle_size, class_num, model_name, filename)
    ensure_dir_exists(path)
    test_net.load_state_dict(torch.load(path))
    test_loader = datasets
    labels_all, f1_activations, f2_activations = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
    dim = middle_size
    indices = torch.randperm(dim)
    perm = torch.randperm(dim)
    permnew = torch.randperm(dim)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            labels_all = torch.cat([labels_all, labels], dim=0)
            outputs = test_net(images)
            feature_map = test_net.feature_map
            if model_name == MODEL_NAME[0]:
                f1_activations = torch.cat([f1_activations, feature_map[0][:, indices[:int(dim / 2)]]], dim=0)
                f2_activations = torch.cat([f2_activations, feature_map[0][:, indices[int(dim / 2):]]], dim=0)
            elif model_name == MODEL_NAME[1] or model_name == MODEL_NAME[2] or \
                    model_name == MODEL_NAME[3] or model_name == MODEL_NAME[4] or model_name == MODEL_NAME[5]:
                if shuffle:
                    temp = torch.cat([feature_map[0], feature_map[1]], dim=1)
                    temp_shuffled = temp[:, perm]
                    f1_activations = torch.cat([f1_activations, temp_shuffled[:, permnew[:int(dim / 2)]]], dim=0)
                    f2_activations = torch.cat([f2_activations, temp_shuffled[:, permnew[int(dim / 2):]]], dim=0)
                else:
                    f1_activations = torch.cat([f1_activations, feature_map[0]], dim=0)
                    f2_activations = torch.cat([f2_activations, feature_map[1]], dim=0)
    labels_number = labels_all % 10
    labels_dot = (labels_all / 10).trunc()
    return f1_activations, f2_activations, labels_number, labels_dot, labels_all



def calculate_fid(f_real, f_generated):
    mu_real, mu_generated = np.mean(f_real, axis=0), np.mean(f_generated, axis=0)
    cov_real, cov_generated = np.cov(f_real, rowvar=False), np.cov(f_generated, rowvar=False)
    mu_diff = mu_real - mu_generated
    cov_mean = (cov_real + cov_generated) / 2
    fid = np.dot(mu_diff, mu_diff) + np.trace(cov_real + cov_generated - 2.0 * np.sqrt(cov_mean))
    return fid


def compute_fid_separation_metric(feature_groups):
    n = len(feature_groups)
    all_fids = []
    fid_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            fid_value = calculate_fid(feature_groups[i], feature_groups[j])
            all_fids.append(fid_value)
            fid_matrix[i, j] = fid_value
            fid_matrix[j, i] = fid_value  # FID is symmetric

    return np.mean(all_fids), fid_matrix


def compute_FID(datasets, train_idx, epoch, model_name='', class_num=10, lambda_sparse=0.01, model_config=[], out_dir=''):
    f1_activations, f2_activations, labels_number, labels_dot, labels_all = get_activation(datasets=datasets, train_idx=train_idx,
                                                                                           epoch=epoch, model_name=model_name,
                                                                                           class_num=class_num, shuffle=False,
                                                                                           lambda_sparse=lambda_sparse,
                                                                                           model_config=model_config, out_dir=out_dir)
    unique_labels_number, unique_labels_dot = torch.unique(labels_number), torch.unique(labels_dot)
    f1_activations_by_label_number = [f1_activations[labels_number == label].numpy() for label in unique_labels_number]
    f1_activations_by_label_dot = [f1_activations[labels_dot == label].numpy() for label in unique_labels_dot]
    f2_activations_by_label_number = [f2_activations[labels_number == label].numpy() for label in unique_labels_number]
    f2_activations_by_label_dot = [f2_activations[labels_dot == label].numpy() for label in unique_labels_dot]
    separation_metric1_number, separation_matrix1_number = compute_fid_separation_metric(f1_activations_by_label_number)
    separation_metric1_number = round(separation_metric1_number, 2)
    separation_metric1_dot, separation_matrix1_dot = compute_fid_separation_metric(f1_activations_by_label_dot)
    separation_metric1_dot = round(separation_metric1_dot, 2)
    separation_metric2_number, separation_matrix2_number = compute_fid_separation_metric(f2_activations_by_label_number)
    separation_metric2_number = round(separation_metric2_number, 2)
    separation_metric2_dot, separation_matrix2_dot = compute_fid_separation_metric(f2_activations_by_label_dot)
    separation_metric2_dot = round(separation_metric2_dot, 2)
    fid_all = [separation_metric1_dot, separation_metric1_number, separation_metric2_dot, separation_metric2_number]
    fid_matrix_all = [separation_matrix1_dot, separation_matrix1_number, separation_matrix2_dot, separation_matrix2_number]

    return fid_all, fid_matrix_all
