from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import wandb
import argparse
import numpy as np
import torch
import torch.nn as nn
from models import MLP, TransformerNetwork, TransformerNetworkPCT, shapeClassifier
from data import BasicPointCloudDataset
torch.set_default_dtype(torch.float32)
import torch.nn.functional as F


def test(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    total_K1_loss = 0.0
    total_K2_loss = 0.0
    total_normal_loss = 0.0
    count = 0
    mseLoss = nn.MSELoss()

    with torch.no_grad():
        for batch in dataloader:
            pcl = batch['point_cloud'].to(device).float()
            normals_orig = batch['normal_vec'].to(device).float()
            # info = batch['info']
            # K1_orig = info['k1'].to(device).float()
            # K2_orig = info['k2'].to(device).float()
            output = (model((pcl.permute(0, 2, 1)).unsqueeze(2))).squeeze()
            # output = F.normalize(output, p=2, dim=1)
            # K1 = output[:, 0]
            # K2 = output[:, 1]
            # normals = output[:, 2:]
            normals = output
            # K1_loss = mseLoss(K1_orig, K1)
            # K2_loss = mseLoss(K2_orig, K2)
            # normals_loss = mseLoss(normals_orig, normals)
            normals_loss = torch.min(
                (normals_orig - normals).pow(2).sum(1),  # Case 1: Direct matching
                (normals_orig + normals).pow(2).sum(1)  # Case 2: Inverted matching
            ).mean()

            # total_K1_loss += K1_loss.item()
            # total_K2_loss += K2_loss.item()
            total_normal_loss += normals_loss.item()
            count += 1

    batch_K1_loss = (total_K1_loss / count)
    batch_K2_loss = (total_K2_loss / count)
    batch_normal_loss = (total_normal_loss / count)

    return batch_K1_loss, batch_K2_loss, batch_normal_loss



def train_and_test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.use_wandb:
        wandb.login(key="ed8e8f26d1ee503cda463f300a605cb35e75ad23")
        wandb.init(project=args.wandb_proj, name=args.exp_name)

    print(device)
    print(args)
    num_epochs = args.epochs
    learning_rate = args.lr
    if args.output_dim==5:
        train_dataset = BasicPointCloudDataset(file_path="train_surfaces_05X05.h5" , args=args)
        test_dataset = BasicPointCloudDataset(file_path='test_surfaces_05X05.h5' , args=args)
    elif args.output_dim==4:
        train_dataset = BasicPointCloudDataset(file_path="train_surfaces_05X05_no_edge.h5" , args=args)
        test_dataset = BasicPointCloudDataset(file_path="test_surfaces_05X05_no_edge.h5" , args=args)
    else:
        train_dataset = BasicPointCloudDataset(file_path="train_surfaces_05X05_no_edge.h5" , args=args)
        test_dataset = BasicPointCloudDataset(file_path='test_surfaces_05X05_no_edge.h5' , args=args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = shapeClassifier(args).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Num of parameters in NN: {num_params}')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # milestones = np.linspace(args.lr_jumps,num_epochs,num_epochs//args.lr_jumps)
    milestones = [args.lr_jumps * (i) for i in range(1,num_epochs//args.lr_jumps + 1)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    mseLoss = nn.MSELoss()
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_K1_loss = 0.0
        total_K2_loss = 0.0
        total_normal_loss = 0.0
        count = 0
        # Use tqdm to create a progress bar for the training loop
        with tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False) as tqdm_bar:
            for batch in tqdm_bar:
                pcl = batch['point_cloud'].to(device).float()  # Explicitly convert to float32
                normals_orig = batch['normal_vec'].to(device).float()
                # info = batch['info']
                # K1_orig = info['k1'].to(device).float()  # Convert to float32
                # K2_orig = info['k2'].to(device).float()
                output = (model((pcl.permute(0, 2, 1)).unsqueeze(2))).squeeze()
                # output = F.normalize(output, p=2, dim=1)

                # K1 = output[:, 0]
                # K2 = output[:, 1]
                # normals = output[:, 2:]
                normals = output
                # K1_loss = mseLoss(K1_orig, K1)
                # K2_loss = mseLoss(K2_orig, K2)
                # normals_loss = mseLoss(normals_orig, normals)
                normals_loss = torch.min(
                    (normals_orig - normals).pow(2).sum(1),
                    (normals_orig + normals).pow(2).sum(1)
                ).mean()

                new_awesome_loss = normals_loss
                # new_awesome_loss = K1_loss + K2_loss + normals_loss
                # new_awesome_loss = K1_loss + K2_loss
                optimizer.zero_grad()
                new_awesome_loss.backward()
                optimizer.step()

                current_lr = optimizer.param_groups[0]['lr']

                # total_K1_loss += K1_loss.item()
                # total_K2_loss += K2_loss.item()
                total_normal_loss += normals_loss.item()

                count = count + 1

        train_K1_loss = (total_K1_loss / count)
        train_K2_loss = (total_K2_loss / count)
        train_normal_loss = (total_normal_loss / count)
        print(f'train_K1_loss: {train_K1_loss}')
        print(f'train_K2_loss: {train_K2_loss}')
        print(f'train_normal_loss: {train_normal_loss}')
        test_K1_loss, test_K2_loss, test_normal_loss = test(model, test_dataloader, device)

        print(f'test_K1_loss: {test_K1_loss}')
        print(f'test_K2_loss: {test_K2_loss}')
        print(f'test_normal_loss: {test_normal_loss}')
        print(f'LR: {current_lr}')
        scheduler.step()

        if args.use_wandb:
            wandb.log({"epoch": epoch, "train_K1_loss": train_K1_loss, "train_K2_loss": train_K2_loss, "train_normal_loss": train_normal_loss,
                       "test_K1_loss": test_K1_loss, "test_K2_loss": test_K2_loss, "test_normal_loss": test_normal_loss})
    return model

def configArgsPCT():
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--wandb_proj', type=str, default='MLP-Contrastive-Ablation', metavar='N',
                        help='Name of the wandb project name to upload the run data')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--use_wandb', type=int, default=1, metavar='N',
                        help='use angles in learning ')
    parser.add_argument('--contr_margin', type=float, default=1.0, metavar='N',
                        help='margin used for contrastive loss')
    parser.add_argument('--use_lap_reorder', type=int, default=1, metavar='N',
                        help='reorder points by laplacian order ')
    parser.add_argument('--lap_eigenvalues_dim', type=int, default=0, metavar='N',
                        help='use eigenvalues in as input')
    parser.add_argument('--use_second_deg', type=int, default=1, metavar='N',
                        help='use second degree embedding ')
    parser.add_argument('--lpe_normalize', type=int, default=1, metavar='N',
                        help='use normalized laplacian')
    parser.add_argument('--std_dev', type=float, default=0.05, metavar='N',
                        help='amount of noise to add to data')
    parser.add_argument('--max_curve_diff', type=float, default=2, metavar='N',
                        help='max difference in curvature for contrastive loss')
    parser.add_argument('--min_curve_diff', type=float, default=0.05, metavar='N',
                        help='min difference in curvature for contrastive loss')
    parser.add_argument('--clip', type=float, default=0.25, metavar='N',
                        help='clip noise')
    parser.add_argument('--contr_loss_weight', type=float, default=0.1, metavar='N',
                        help='weight of contrastive loss')
    parser.add_argument('--lpe_dim', type=int, default=0, metavar='N',
                        help='laplacian positional encoding amount of eigens to take')
    parser.add_argument('--use_xyz', type=int, default=1, metavar='N',
                        help='use xyz coordinates as part of input')
    parser.add_argument('--classification', type=int, default=1, metavar='N',
                        help='use classification loss')
    parser.add_argument('--rotate_data', type=int, default=1, metavar='N',
                        help='use rotated data')
    parser.add_argument('--cube', type=int, default=0, metavar='N',
                        help='Normalize data into 1 cube')
    parser.add_argument('--num_neurons_per_layer', type=int, default=64, metavar='N',
                        help='how many neurons per layer to use')
    parser.add_argument('--num_mlp_layers', type=int, default=5, metavar='N',
                        help='how many mlp layers to use')
    parser.add_argument('--output_dim', type=int, default=5, metavar='N',
                        help='how many labels are used')
    parser.add_argument('--lr_jumps', type=int, default=15, metavar='N',
                        help='Lower lr *0.1 every amount of jumps')
    parser.add_argument('--sampled_points', type=int, default=20, metavar='N',
                        help='How many points where sampled around centroid')
    args = parser.parse_args()
    return args
def testPretrainedModel(args, model=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_dataset = BasicPointCloudDataset(file_path='test_surfaces_05X05.h5', args=args)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Num of parameters in NN: {num_params}')
    # Set the model to evaluation mode
    model.eval()
    count =0
    total_acc_loss = 0.0
    label_correct = {label: 0 for label in range(5)}
    label_total = {label: 0 for label in range(5)}
    wrong_preds = {label: [] for label in range(5)}
    wrong_idx = {label: [] for label in range(5)}
    wrong_pcl = {label: [] for label in range(5)}
    wrong_pred_class = {label: [] for label in range(5)}
    wrong_K1_values = {label: [] for label in range(5)}
    wrong_K2_values = {label: [] for label in range(5)}

    with torch.no_grad():
        for batch in test_dataloader:
            pcl, info = batch['point_cloud'].to(device), batch['info']
            label = info['class'].to(device).long()
            output = model((pcl.permute(0, 2, 1)).unsqueeze(2))
            output = (output[:,:5]).squeeze()
            preds = output.max(dim=1)[1]
            total_acc_loss += torch.mean((preds == label).float()).item()

            # Collect data for wrong predictions
            for i, (pred, actual_label) in enumerate(zip(preds, label.cpu().numpy())):
                if pred != actual_label:
                    wrong_preds[actual_label].append(pred.item())
                    wrong_idx[actual_label].append(info['idx'][i].item())
                    wrong_pcl[actual_label].append(pcl[i,:,:])
                    wrong_pred_class[actual_label].append(preds[i])
                    wrong_K1_values[actual_label].append((info['k1'][i].item()))
                    wrong_K2_values[actual_label].append((info['k2'][i].item()))

            count += 1

            # Update per-label statistics
            for label_name in range(5):
                correct_mask = (preds == label_name) & (label == label_name)
                label_correct[label_name] += correct_mask.sum().item()
                label_total[label_name] += (label == label_name).sum().item()

    label_accuracies = {
        label: label_correct[label] / label_total[label]
        for label in range(5)
        if label_total[label] != 0
    }
    for label, accuracy in label_accuracies.items():
        print(f"Accuracy for label {label}: {accuracy:.4f}")

    # for label in range(4):
    for label in range(5):
        if len(wrong_preds[label]) > 0:
            print(f"Label {label}:")
            print(f"  - Most frequent wrong prediction: {max(wrong_preds[label], key=wrong_preds[label].count)}")
            print(f"  - Average K1 for wrong predictions: {np.mean(wrong_K1_values[label])}")
            print(f"  - Average K2 for wrong predictions: {np.mean(wrong_K2_values[label])}")
            print(f"  - median K1 for wrong predictions: {np.median(wrong_K1_values[label])}")
            print(f"  - median K2 for wrong predictions: {np.median(wrong_K2_values[label])}")
            print(f"+++++")
            argmax_K1_index = (np.argmax(np.abs(wrong_K1_values[label])))
            print(f"  - biggest abs wrong K1 pcl idx: {wrong_idx[label][argmax_K1_index]}")
            print(f"  - biggest abs wrong K1 pcl val: {wrong_K1_values[label][argmax_K1_index]}")
            np.save(f"{label}_max_K1_pcl_{wrong_pred_class[label][argmax_K1_index]}.npy", (wrong_pcl[label][argmax_K1_index]).cpu().numpy() )

            argmin_K1_index = (np.argmin(np.abs(wrong_K1_values[label])))
            print(f"  - smallest abs wrong K1 pcl idx: {wrong_idx[label][argmin_K1_index]}")
            print(f"  - smallest abs wrong K1 pcl val: {wrong_K1_values[label][argmin_K1_index]}")
            np.save(f"{label}_min_K1_pcl_{wrong_pred_class[label][argmin_K1_index]}.npy", (wrong_pcl[label][argmin_K1_index]).cpu().numpy())

            argmax_K2_index = (np.argmax(np.abs(wrong_K2_values[label])))
            print(f"  - biggest abs wrong K2 pcl idx: {wrong_idx[label][argmax_K2_index]}")
            print(f"  - biggest abs wrong K2 pcl val: {wrong_K2_values[label][argmax_K2_index]}")
            np.save(f"{label}_max_K2_pcl_{wrong_pred_class[label][argmax_K2_index]}.npy", (wrong_pcl[label][argmax_K2_index]).cpu().numpy())

            argmin_K2_index = (np.argmin(np.abs(wrong_K2_values[label])))
            print(f"  - smallest abs wrong K2 pcl idx: {wrong_idx[label][argmin_K2_index]}")
            print(f"  - smallest abs wrong K2 pcl val: {wrong_K2_values[label][argmin_K2_index]}")
            np.save(f"{label}_min_K2_pcl_{wrong_pred_class[label][argmin_K2_index]}.npy", (wrong_pcl[label][argmin_K2_index]).cpu().numpy())

import cProfile
import pstats
if __name__ == '__main__':
    args = configArgsPCT()
    model = train_and_test(args)
    torch.save(model.state_dict(), f'{args.exp_name}.pt')
    # testPretrainedModel(args, model=model.to('cuda'))


