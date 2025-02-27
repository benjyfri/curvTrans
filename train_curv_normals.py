from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import wandb
import argparse
import torch
import torch.nn as nn
from models import shapeClassifier
from data import BasicPointCloudDataset
import torch.nn.functional as F

def cos_angle(v1, v2):
    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)


def test_curvatues(model, dataloader, device, args):
    model.eval()
    total_H_loss = 0.0
    total_K_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            pcl, info = batch['point_cloud'].to(device), batch['info']
            normals_orig = batch['normal_vec'].to(device).float()
            label = info['class'].to(device).long()
            k1 = info['k1'].to(device).float()
            k2 = info['k2'].to(device).float()

            K_GT = k1 * k2
            H_GT = torch.abs(0.5 * (k1 + k2))

            output = model((pcl.permute(0, 2, 1)).unsqueeze(2)).squeeze()

            if args.output_dim==2:
                cur_K = output[:, 0]
                cur_H = torch.abs(output[:, 1])
            if args.output_dim == 6:
                cur_K = output[:, 4]
                cur_H = torch.abs(output[:, 5])

            # Compute rectified error
            K_error = (cur_K - K_GT) / torch.max(K_GT.abs(), torch.tensor(1.0, device=device))
            H_error = (cur_H - H_GT) / torch.max(H_GT.abs(), torch.tensor(1.0, device=device))

            # Compute loss using mean-square during training and RMS for evaluation
            K_cur_loss = torch.sqrt(torch.mean(K_error ** 2))
            H_cur_loss = torch.sqrt(torch.mean(H_error ** 2))

            total_H_loss += H_cur_loss.item()
            total_K_loss += K_cur_loss.item()
            count += 1

    test_H_loss = total_H_loss / count
    test_K_loss = total_K_loss / count
    return test_H_loss, test_K_loss


def test(model, dataloader, device, args):
    model.eval()
    total_loss = 0.0
    total_acc_loss = 0.0
    total_normal_loss = 0.0
    total_H_loss = 0.0
    total_K_loss = 0.0
    count = 0
    label_correct = {label: 0 for label in range(4)}
    label_total = {label: 0 for label in range(4)}
    criterion = nn.CrossEntropyLoss(reduction='mean')
    mseLoss = nn.MSELoss()
    with torch.no_grad():
        for batch in dataloader:
            pcl, info = batch['point_cloud'].to(device), batch['info']
            normals_orig = (batch['normal_vec'].to(device).float())
            label = info['class'].to(device).long()
            k1 = info['k1'].to(device).float()
            k2 = info['k2'].to(device).float()
            K_GT = k1 * k2
            H_GT = torch.abs(0.5 * (k1 + k2))
            output = (model((pcl.permute(0, 2, 1)).unsqueeze(2))).squeeze()

            # orig_classification = output[:, :4]
            # normals = (output[:, 4:])
            # classification_loss = criterion(orig_classification, label)
            # normals = (output)
            # normals_loss = torch.min(
            #     (normals_orig - normals).pow(2).sum(1),
            #     (normals_orig + normals).pow(2).sum(1)
            # ).mean()
            # normals_loss = mseLoss(normals_orig, normals)
            # preds = orig_classification.max(dim=1)[1]
            # total_acc_loss += torch.mean((preds == label).float()).item()
            # total_normal_loss += normals_loss.item()
            # total_loss += classification_loss.item()

            orig_classification = output[:, :4]
            classification_loss = criterion(orig_classification, label)
            preds = orig_classification.max(dim=1)[1]
            total_acc_loss += torch.mean((preds == label).float()).item()
            cur_K = output[:, 4]
            cur_H = torch.abs(output[:, 5])
            K_cur_loss = mseLoss(cur_K, K_GT)
            H_cur_loss = mseLoss(cur_H, H_GT)
            total_H_loss += H_cur_loss.item()
            total_K_loss += K_cur_loss.item()

            count += 1
            for label_name in range(4):
                correct_mask = (preds == label_name) & (label == label_name)
                label_correct[label_name] += correct_mask.sum().item()
                label_total[label_name] += (label == label_name).sum().item()

    # Overall accuracy
    test_cls_loss = (total_loss / (count))
    test_acc = (total_acc_loss / (count))
    label_accuracies = {label: label_correct[label] / label_total[label] if label_total[label] != 0 else 0.0
                        for label in range(4)}
    # normal_loss = (total_normal_loss / count)
    #
    # return normal_loss, test_cls_loss, test_acc, label_accuracies
    test_H_loss = (total_H_loss / (count))
    test_k_loss = (total_K_loss / (count))
    return test_H_loss, test_k_loss, test_acc, label_accuracies

def train_and_test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.use_wandb:
        wandb.login(key="ed8e8f26d1ee503cda463f300a605cb35e75ad23")
        wandb.init(project=args.wandb_proj, name=args.exp_name)

    print(device)
    print(args)
    num_epochs = args.epochs
    learning_rate = args.lr
    if args.output_dim == 5:
        train_dataset = BasicPointCloudDataset(file_path="train_surfaces_05X05.h5", args=args)
        test_dataset = BasicPointCloudDataset(file_path='test_surfaces_05X05.h5', args=args)
    elif args.output_dim == 4:
        train_dataset = BasicPointCloudDataset(file_path="train_surfaces_05X05_no_edge.h5", args=args)
        test_dataset = BasicPointCloudDataset(file_path="test_surfaces_05X05_no_edge.h5", args=args)
    elif args.output_dim == 3:
        train_dataset = BasicPointCloudDataset(file_path="train_surfaces_05X05_only_bumps.h5", args=args)
        test_dataset = BasicPointCloudDataset(file_path="test_surfaces_05X05_only_bumps.h5", args=args)
    else:
        train_dataset = BasicPointCloudDataset(file_path="train_surfaces_05X05_no_edge.h5", args=args)
        test_dataset = BasicPointCloudDataset(file_path='test_surfaces_05X05_no_edge.h5', args=args)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = shapeClassifier(args).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Num of parameters in NN: {num_params}')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    milestones = [args.lr_jumps * (i) for i in range(1, num_epochs // args.lr_jumps + 1)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    mseLoss = nn.MSELoss()
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_train_loss = 0.0
        total_normal_loss = 0.0
        total_H_loss = 0.0
        total_K_loss = 0.0
        total_train_acc_loss = 0.0
        count = 0
        # Use tqdm to create a progress bar for the training loop
        with tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False) as tqdm_bar:
            for batch in tqdm_bar:
                pcl, info = batch['point_cloud'].to(device), batch['info']
                # normals_orig = (batch['normal_vec'].to(device).float())
                label = info['class'].to(device).long()
                k1 = info['k1'].to(device).float()
                k2 = info['k2'].to(device).float()
                K_GT = k1 * k2
                H_GT = torch.abs(0.5 * (k1 + k2))
                output = (model((pcl.permute(0, 2, 1)).unsqueeze(2))).squeeze()

                # orig_classification = output[:, :4]
                # normals = (output[:, 4:])
                # normals = (output)
                # # classification_loss = criterion(orig_classification, label)
                # normals_loss = torch.min(
                #     (normals_orig - normals).pow(2).sum(1),
                #     (normals_orig + normals).pow(2).sum(1)
                # ).mean()
                # normals_loss = mseLoss(normals_orig, normals)

                # new_awesome_loss = classification_loss + normals_loss
                # new_awesome_loss = normals_loss
                orig_classification = output[:, :4]
                classification_loss = criterion(orig_classification, label)
                cur_K = output[:,4]
                cur_H = torch.abs(output[:,5])
                K_cur_loss = mseLoss(cur_K, K_GT)
                H_cur_loss = mseLoss(cur_H, H_GT)
                new_awesome_loss = K_cur_loss + H_cur_loss + classification_loss
                optimizer.zero_grad()
                new_awesome_loss.backward()
                optimizer.step()

                current_lr = optimizer.param_groups[0]['lr']

                total_train_loss += classification_loss.item()
                # total_normal_loss += normals_loss.item()
                total_H_loss += H_cur_loss.item()
                total_K_loss += K_cur_loss.item()
                preds = orig_classification.max(dim=1)[1]
                total_train_acc_loss += torch.mean((preds == label).float()).item()

                count = count + 1

        train_acc = (total_train_acc_loss / (count))
        train_classification_loss = (total_train_loss/ (count))
        # train_normal_loss = (total_normal_loss / count)
        train_H_loss = (total_H_loss / count)
        train_K_loss = (total_K_loss / count)
        # test_normal_loss,test_classification_loss, test_acc, test_label_accuracies= test(model, test_dataloader, device, args)
        # test_H_loss, test_k_loss = test(model, test_dataloader, device, args)
        test_H_loss, test_k_loss, test_acc, test_label_accuracies = test(model, test_dataloader, device, args)
        print(f'LR: {current_lr}')
        dict_of_vals = {"epoch": epoch, "train_H_loss": train_H_loss, "train_K_loss": train_K_loss,
               "test_H_loss": test_H_loss, "test_k_loss": test_k_loss,"train_acc":train_acc, "train_classification_loss":train_classification_loss,
                        "test_acc":test_acc}
        print(dict_of_vals)
        for key in test_label_accuracies:
            print("label_" + str(key), ":", test_label_accuracies[key])

        scheduler.step()

        if args.use_wandb:
            wandb.log(dict_of_vals)
            for key in test_label_accuracies:
                wandb.log({"epoch": epoch, "label_" + str(key): test_label_accuracies[key]})
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


if __name__ == '__main__':
    # args = configArgsPCT()
    # model = train_and_test(args)
    # torch.save(model.state_dict(), f'{args.exp_name}.pt')

    args = configArgsPCT()
    print(args.exp_name)
    test_dataset = BasicPointCloudDataset(file_path='test_surfaces_05X05_no_edge.h5', args=args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = shapeClassifier(args)
    model.load_state_dict(torch.load(f'{args.exp_name}.pt',weights_only=True))
    model.to('cuda')
    output = test_curvatues(model, test_dataloader, 'cuda', args)
    print(output)

    args = configArgsPCT()
    args.exp_name = "f_norm"
    args.output_dim = 2
    print(args.exp_name)
    test_dataset = BasicPointCloudDataset(file_path='test_surfaces_05X05_no_edge.h5', args=args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = shapeClassifier(args)
    model.load_state_dict(torch.load(f'{args.exp_name}.pt',weights_only=True))
    model.to('cuda')
    output = test_curvatues(model, test_dataloader, 'cuda', args)
    print(output)


