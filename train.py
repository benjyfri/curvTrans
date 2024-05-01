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
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns


def test(model, dataloader, loss_function, device, args):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_acc_loss = 0.0
    count = 0
    label_correct = {label: 0 for label in range(args.output_dim)}
    label_total = {label: 0 for label in range(args.output_dim)}

    with torch.no_grad():
        for batch in dataloader:
            pcl, info = batch['point_cloud'].to(device), batch['info']
            label = info['class'].to(device).long()
            if args.contrastive_mid_layer:
                output,_ = model((pcl.permute(0, 2, 1)).unsqueeze(2))
            else:
                output = model((pcl.permute(0, 2, 1)).unsqueeze(2))
            loss = loss_function(output, label)
            preds = output.max(dim=1)[1]
            total_acc_loss += torch.mean((preds == label).float()).item()
            total_loss += loss.item()
            count += 1
            for label_name in range(args.output_dim):
                correct_mask = (preds == label_name) & (label == label_name)
                label_correct[label_name] += correct_mask.sum().item()
                label_total[label_name] += (label == label_name).sum().item()

    # Overall accuracy
    test_acc = (total_acc_loss / (count))
    label_accuracies = {label: label_correct[label] / label_total[label] if label_total[label] != 0 else 0.0
                        for label in range(args.output_dim)}
    average_loss = total_loss / (count)

    return average_loss, test_acc, label_accuracies



def train_and_test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.use_wandb:
        wandb.login(key="ed8e8f26d1ee503cda463f300a605cb35e75ad23")
        wandb.init(project="Curvature-transformer-POC-fixedHK", name=args.exp_name)

    print(device)
    print(args)
    num_epochs = args.epochs
    learning_rate = args.lr

    # Create instances for training and testing datasets
    if args.sampled_points==20:
        train_dataset = BasicPointCloudDataset(file_path="train_surfaces.h5" , args=args)
        test_dataset = BasicPointCloudDataset(file_path='test_surfaces.h5' , args=args)
    if args.sampled_points==40:
        train_dataset = BasicPointCloudDataset(file_path="train_surfaces_40_stronger_boundaries.h5" , args=args)
        test_dataset = BasicPointCloudDataset(file_path='test_surfaces_40_stronger_boundaries.h5' , args=args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = shapeClassifier(args).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Num of parameters in NN: {num_params}')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # milestones = np.linspace(args.lr_jumps,num_epochs,num_epochs//args.lr_jumps)
    milestones = [args.lr_jumps * (i) for i in range(1,num_epochs//args.lr_jumps + 1)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    tripletMarginLoss = nn.TripletMarginLoss(margin=2.0)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    mseLoss = nn.MSELoss()
    contr_loss_weight = args.contr_loss_weight
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_train_loss = 0.0
        total_train_contrastive_loss = 0.0
        total_train_contrastive_positive_loss = 0.0
        total_train_contrastive_negative_loss = 0.0
        total_train_acc_loss = 0.0
        count = 0
        # Use tqdm to create a progress bar for the training loop
        with tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False) as tqdm_bar:
            for batch in tqdm_bar:
                pcl, info = batch['point_cloud'].to(device), batch['info']

                label = info['class'].to(device).long()
                if args.contrastive:
                    pcl2 = batch['point_cloud2'].to(device)
                    contrastive_point_cloud = batch['contrastive_point_cloud'].to(device)
                    if args.contrastive_mid_layer:
                        output, layer_before_last = model((pcl.permute(0, 2, 1)).unsqueeze(2))
                        loss = criterion(output, label)
                        output_pcl2, layer_before_last_pcl2 = model((pcl2.permute(0, 2, 1)).unsqueeze(2))
                        output_contrastive_pcl, layer_before_last_contrastive_pcl = model((contrastive_point_cloud.permute(0, 2, 1)).unsqueeze(2))
                        loss2 = tripletMarginLoss(layer_before_last, layer_before_last_pcl2, layer_before_last_contrastive_pcl)
                        total_train_contrastive_positive_loss += mseLoss(layer_before_last, layer_before_last_pcl2)
                        total_train_contrastive_negative_loss += mseLoss(layer_before_last, layer_before_last_contrastive_pcl)
                    else:
                        output = model((pcl.permute(0, 2, 1)).unsqueeze(2))
                        orig_classification = output[:,:4]
                        orig_emb = output[:,4:]
                        loss = criterion(orig_classification, label)
                        output_pcl2 = model((pcl2.permute(0, 2, 1)).unsqueeze(2))
                        pos_emb = output_pcl2[:, 4:]
                        output_contrastive_pcl = model((contrastive_point_cloud.permute(0, 2, 1)).unsqueeze(2))
                        neg_emb = output_contrastive_pcl[:, 4:]
                        loss2 = tripletMarginLoss(orig_emb, pos_emb, neg_emb)
                        total_train_contrastive_positive_loss += mseLoss(orig_emb, pos_emb)
                        total_train_contrastive_negative_loss += mseLoss(orig_emb, neg_emb)
                    total_train_contrastive_loss += loss2
                    loss2 = (contr_loss_weight) * loss2
                else:
                    output  = model((pcl.permute(0, 2, 1)).unsqueeze(2))
                    loss = criterion(output, label)
                    loss2 = torch.tensor((0))
                new_awesome_loss = loss + loss2
                # Backward pass and optimization
                optimizer.zero_grad()
                # loss.backward()
                new_awesome_loss.backward()
                optimizer.step()

                current_lr = optimizer.param_groups[0]['lr']

                total_train_loss += loss.item()
                preds = output.max(dim=1)[1]
                total_train_acc_loss += torch.mean((preds == label).float()).item()

                count = count + 1

                tqdm_bar.set_postfix(train_loss=f'{(loss.item() / args.batch_size):.4f}')

        train_loss = (total_train_loss / count)
        contrastive_train_loss = (total_train_contrastive_loss / (count))
        contrastive_positive_train_loss = (total_train_contrastive_positive_loss / (count))
        contrastive_negative_train_loss = (total_train_contrastive_negative_loss / (count))
        print(f'contrastive_train_loss: {contrastive_train_loss}')
        print(f'contrastive_positive_train_loss: {contrastive_positive_train_loss}')
        print(f'contrastive_negative_train_loss: {contrastive_negative_train_loss}')
        acc_train = (total_train_acc_loss / (count))


        test_loss, acc_test, label_accuracies = test(model, test_dataloader, criterion, device, args)
        scheduler.step()
        print(f'LR: {current_lr}')

        print({"epoch": epoch, "train_loss": train_loss ,"test_loss": test_loss, "acc_train": acc_train, "acc_test": acc_test})
        for key in label_accuracies:
            print("label_" + str(key), ":", label_accuracies[key])
        if args.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss ,"test_loss": test_loss, "acc_train": acc_train, "acc_test": acc_test})
            for key in label_accuracies:
                wandb.log({"epoch": epoch, "label_"+str(key) : label_accuracies[key]})

    return model

def configArgsPCT():
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=512, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--use_wandb', type=int, default=0, metavar='N',
                        help='use angles in learning ')
    parser.add_argument('--use_second_deg', type=int, default=0, metavar='N',
                        help='use second degree embedding ')
    parser.add_argument('--lpe_normalize', type=int, default=0, metavar='N',
                        help='use PCT transformer version')
    parser.add_argument('--std_dev', type=float, default=0, metavar='N',
                        help='amount of noise to add to data')
    parser.add_argument('--contr_loss_weight', type=float, default=1.0, metavar='N',
                        help='weight of contrastive loss')
    parser.add_argument('--lpe_dim', type=int, default=3, metavar='N',
                        help='laplacian positional encoding amount of eigens to take')
    parser.add_argument('--use_xyz', type=int, default=1, metavar='N',
                        help='use xyz coordinates as part of input')
    parser.add_argument('--contrastive', type=int, default=0, metavar='N',
                        help='use contrastive loss')
    parser.add_argument('--contrastive_mid_layer', type=int, default=0, metavar='N',
                        help='use contrastive loss with middle layer (one before last)')
    parser.add_argument('--rotate_data', type=int, default=0, metavar='N',
                        help='use rotated data')
    parser.add_argument('--num_neurons_per_layer', type=int, default=64, metavar='N',
                        help='how many neurons per layer to use')
    parser.add_argument('--num_mlp_layers', type=int, default=4, metavar='N',
                        help='how many mlp layers to use')
    parser.add_argument('--output_dim', type=int, default=4, metavar='N',
                        help='how many labels are used')
    parser.add_argument('--lr_jumps', type=int, default=50, metavar='N',
                        help='Lower lr *0.1 every amount of jumps')
    parser.add_argument('--sampled_points', type=int, default=40, metavar='N',
                        help='How many points where sampled around centroid')
    args = parser.parse_args()
    return args
def testPretrainedModel(args, model=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.sampled_points == 20:
        test_dataset = BasicPointCloudDataset(file_path='test_surfaces.h5', args=args)
    elif args.sampled_points == 40:
        test_dataset = BasicPointCloudDataset(file_path='test_surfaces_40_stronger_boundaries.h5', args=args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    if model is None:
        model = MLP(input_size=36 * (args.sampled_points + 1), num_layers=args.num_mlp_layers,
                    num_neurons_per_layer=args.num_neurons_per_layer, output_size=args.output_dim).to(device)

        # Load the saved state dictionary
        model_path = r"best.pt"  # Update with the path to your saved model
        model.load_state_dict(torch.load(model_path))
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Num of parameters in NN: {num_params}')
    # Set the model to evaluation mode
    model.eval()
    count =0
    total_acc_loss = 0.0
    label_correct = {label: 0 for label in range(args.output_dim)}
    label_total = {label: 0 for label in range(args.output_dim)}
    wrong_preds = {label: [] for label in range(args.output_dim)}
    wrong_H_values = {label: [] for label in range(args.output_dim)}
    wrong_K_values = {label: [] for label in range(args.output_dim)}
    wrong_predictions_stats = {}  # Store statistics for wrong predictions

    with torch.no_grad():
        for batch in test_dataloader:
            pcl, info = batch['point_cloud'].to(device), batch['info']
            label = info['class'].to(device).long()
            # torch.save(pcl[0].permute(1,0).unsqueeze(1).unsqueeze(0), 'pcl.pt')
            if args.contrastive_mid_layer:
                output, _ = model((pcl.permute(0, 2, 1)).unsqueeze(2))
            else:
                output = model((pcl.permute(0, 2, 1)).unsqueeze(2))
            preds = output.max(dim=1)[1]

            if args.output_dim == 4:
                preds = output.max(dim=1)[1]
                total_acc_loss += torch.mean((preds == label).float()).item()

                # Collect data for wrong predictions
                for i, (pred, actual_label) in enumerate(zip(preds, label.cpu().numpy())):
                    if pred != actual_label:
                        wrong_preds[actual_label].append(pred.item())
                        wrong_H_values[actual_label].append(info['H'][i].item())
                        wrong_K_values[actual_label].append(info['K'][i].item())

            count += 1

            if args.output_dim == 4:
                # Update per-label statistics
                for label_name in range(args.output_dim):
                    correct_mask = (preds == label_name) & (label == label_name)
                    label_correct[label_name] += correct_mask.sum().item()
                    label_total[label_name] += (label == label_name).sum().item()

    if args.output_dim == 4:
        label_accuracies = {
            label: label_correct[label] / label_total[label]
            for label in range(args.output_dim)
            if label_total[label] != 0
        }
        for label, accuracy in label_accuracies.items():
            print(f"Accuracy for label {label}: {accuracy:.4f}")
    for label in range(args.output_dim):
        if len(wrong_preds[label]) > 0:
            print(f"Label {label}:")
            print(f"  - Most frequent wrong prediction: {max(wrong_preds[label], key=wrong_preds[label].count)}")
            print(f"  - Average H for wrong predictions: {np.mean(wrong_H_values[label])}")
            print(f"  - Average K for wrong predictions: {np.mean(wrong_K_values[label])}")
            print(f"  - median H for wrong predictions: {np.median(wrong_H_values[label])}")
            print(f"  - median K for wrong predictions: {np.median(wrong_K_values[label])}")

    # return test_acc, label_accuracies, wrong_predictions_stats

if __name__ == '__main__':
    args = configArgsPCT()
    model = train_and_test(args)
    # model = input_visualized_importance()
    testPretrainedModel(args, model=model.to('cuda'))
    torch.save(model.state_dict(), f'{args.exp_name}.pt')
    # Compute input saliency
    # input_visualized_importance()

