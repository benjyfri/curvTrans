from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import wandb
import argparse
import numpy as np
import torch
import torch.nn as nn
from models import MLP, TransformerNetwork, TransformerNetworkPCT
from data import PointCloudDataset


def test(model, dataloader, loss_function, device, args):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_acc_loss = 0.0
    total_H_loss = 0.0  # Loss for H
    total_K_loss = 0.0  # Loss for K
    count = 0
    label_correct = {label: 0 for label in range(args.output_dim)}
    label_total = {label: 0 for label in range(args.output_dim)}

    with torch.no_grad():
        for batch in dataloader:
            data, lpe, info, pe = batch['point_cloud'].to(device), batch['lpe'].to(device),batch['info'], batch['pe'].to(device)
            if args.output_dim == 2:
                H = info['H'].to(device).float()
                K = info['K'].to(device).float()
                label = torch.stack([H, K], dim=1)
            if args.output_dim == 4:
                label = info['class'].to(device).long()
            if args.use_second_deg:
                x, y, z = data.unbind(dim=2)
                data = torch.stack([x ** 2, x * y, x * z, y ** 2, y * z, z ** 2, x, y, z], dim=2)
            if args.lpe_dim != 0:
                data = torch.cat([data, lpe], dim=2).to(device)
            if args.PE_dim != 0:
                data = torch.cat([data, pe], dim=2).to(device)
            if args.use_xyz == 0:
                data = data[:, :, 3:]
            data = data.permute(0, 2, 1)

            output = model(data)
            if args.output_dim == 2:
                # Calculate and log loss for H and K
                H_loss = loss_function(output[:, 0], H)  # Loss for H
                K_loss = loss_function(output[:, 1], K)  # Loss for K
                total_H_loss += H_loss.item()
                total_K_loss += K_loss.item()
                loss = H_loss + K_loss
            else:
                loss = loss_function(output, label)
            if args.output_dim == 4:
                preds = output.max(dim=1)[1]
                total_acc_loss += torch.mean((preds == label).float()).item()
            total_loss += loss.item()
            count += 1

            if args.output_dim == 4:
                # Update per-label statistics
                for label_name in range(args.output_dim):
                    correct_mask = (preds == label_name) & (label == label_name)
                    label_correct[label_name] += correct_mask.sum().item()
                    label_total[label_name] += (label == label_name).sum().item()

    # Overall accuracy
    if args.output_dim == 4:
        test_acc = (total_acc_loss / (count))
        label_accuracies = {label: label_correct[label] / label_total[label] if label_total[label] != 0 else 0.0
                            for label in range(args.output_dim)}
    if args.output_dim == 2:
        test_acc = 0
        # Calculate average loss for H and K
        avg_H_loss = total_H_loss / count
        avg_K_loss = total_K_loss / count
        label_accuracies = {'H_loss': avg_H_loss, 'K_loss': avg_K_loss}
    average_loss = total_loss / (count * args.batch_size)

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
        train_dataset = PointCloudDataset(file_path="train_surfaces.h5" , args=args)
        test_dataset = PointCloudDataset(file_path='test_surfaces.h5' , args=args)
    if args.sampled_points==40:
        train_dataset = PointCloudDataset(file_path="train_surfaces_40_stronger_boundaries.h5" , args=args)
        test_dataset = PointCloudDataset(file_path='test_surfaces_40_stronger_boundaries.h5' , args=args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    input_dim = 0
    if args.use_xyz:
        input_dim = 3
    if args.use_second_deg:
        input_dim = 9
    if args.lpe_dim!=0:
        input_dim = input_dim + (args.lpe_dim)
    if args.PE_dim!=0:
        input_dim = input_dim + (args.PE_dim *3 *2)
    print(f'input size is {input_dim}')
    if args.use_pct:
        model = TransformerNetworkPCT(input_dim=input_dim, output_dim=args.output_dim, num_heads=args.num_of_heads, num_layers=args.num_of_attention_layers, att_per_layer=4).to(device)
    elif args.use_mlp:
        model = MLP(input_size= input_dim * (args.sampled_points + 1), num_layers=args.num_mlp_layers, num_neurons_per_layer=args.num_neurons_per_layer, output_size=args.output_dim).to(device)
    else:
        model = TransformerNetwork(input_dim=input_dim, output_dim=args.output_dim, num_heads=args.num_of_heads, num_layers=args.num_of_attention_layers).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Num of parameters in NN: {num_params}')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    milestones = np.linspace(args.lr_jumps,num_epochs,num_epochs//args.lr_jumps)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    if args.output_dim == 4:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        criterion = nn.MSELoss()
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_train_loss = 0.0
        total_train_acc_loss = 0.0
        total_H_loss = 0.0  # Loss for H
        total_K_loss = 0.0  # Loss for K
        count = 0
        # Use tqdm to create a progress bar for the training loop
        with tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False) as tqdm_bar:
            for batch in tqdm_bar:
                data, lpe, info, pe = batch['point_cloud'].to(device), batch['lpe'].to(device),batch['info'], batch['pe'].to(device)
                if args.output_dim ==2:
                    H = info['H'].to(device).float()
                    K = info['K'].to(device).float()
                    label = torch.stack([H,K],dim=1)
                if args.output_dim == 4:
                    label = info['class'].to(device).long()
                if args.use_second_deg:
                    x, y, z = data.unbind(dim=2)
                    data = torch.stack([x ** 2, x * y, x * z, y ** 2, y * z, z ** 2, x, y, z], dim=2)
                if args.lpe_dim != 0:
                    data = torch.cat([data, lpe], dim=2).to(device)
                if args.use_xyz==0:
                    data = data[:,:,3:]
                if args.PE_dim != 0:
                    data = torch.cat([data, pe], dim=2).to(device)

                data =  data.permute(0, 2, 1)
                output = model(data)
                if args.output_dim == 2:
                    # Calculate and log loss for H and K
                    H_loss = criterion(output[:, 0], H)  # Loss for H
                    K_loss = criterion(output[:, 1], K)  # Loss for K
                    total_H_loss += H_loss.item()
                    total_K_loss += K_loss.item()
                    loss = H_loss + K_loss
                else:
                    loss = criterion(output, label)


                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current_lr = optimizer.param_groups[0]['lr']

                total_train_loss += loss.item()
                if args.output_dim == 4:
                    preds = output.max(dim=1)[1]
                    total_train_acc_loss += torch.mean((preds == label).float()).item()

                count = count + 1

                tqdm_bar.set_postfix(train_loss=f'{(loss.item() / args.batch_size):.4f}')

        train_loss = (total_train_loss / (args.batch_size * count))

        if args.output_dim == 4:
            acc_train = (total_train_acc_loss / (count))
        if args.output_dim == 2:
            acc_train = 0
            # Calculate average loss for H and K
            avg_H_loss = total_H_loss / count
            avg_K_loss = total_K_loss / count
            if args.use_wandb:
                wandb.log({"epoch": epoch, "train_H_loss": avg_H_loss, "train_K_loss": avg_K_loss})


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
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--use_wandb', type=int, default=0, metavar='N',
                        help='use angles in learning ')
    parser.add_argument('--use_second_deg', type=int, default=0, metavar='N',
                        help='use second degree embedding ')
    parser.add_argument('--lpe_normalize', type=int, default=0, metavar='N',
                        help='use PCT transformer version')
    parser.add_argument('--use_pct', type=int, default=0, metavar='N',
                        help='use PCT transformer version')
    parser.add_argument('--use_mlp', type=int, default=0, metavar='N',
                        help='use PCT transformer version')
    parser.add_argument('--lpe_dim', type=int, default=3, metavar='N',
                        help='laplacian positional encoding amount of eigens to take')
    parser.add_argument('--use_xyz', type=int, default=1, metavar='N',
                        help='use xyz coordinates as part of input')
    parser.add_argument('--num_of_heads', type=int, default=1, metavar='N',
                        help='how many attention heads to use')
    parser.add_argument('--num_neurons_per_layer', type=int, default=64, metavar='N',
                        help='how many neurons per layer to use')
    parser.add_argument('--num_mlp_layers', type=int, default=4, metavar='N',
                        help='how many mlp layers to use')
    parser.add_argument('--num_of_attention_layers', type=int, default=1, metavar='N',
                        help='how many attention layers to use')
    parser.add_argument('--att_per_layer', type=int, default=4, metavar='N',
                        help='how many attention heads in each layer')
    parser.add_argument('--output_dim', type=int, default=4, metavar='N',
                        help='how many labels are used')
    parser.add_argument('--lr_jumps', type=int, default=50, metavar='N',
                        help='Lower lr *0.1 every amount of jumps')
    parser.add_argument('--sampled_points', type=int, default=20, metavar='N',
                        help='How many points where sampled around centroid')
    parser.add_argument('--PE_dim', type=int, default=0, metavar='N',
                        help='Positional embedding size')
    args = parser.parse_args()
    return args
def testPretrainedModel(args, model=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.sampled_points == 20:
        test_dataset = PointCloudDataset(file_path='test_surfaces.h5', args=args)
    elif args.sampled_points == 40:
        test_dataset = PointCloudDataset(file_path='test_surfaces_40.h5', args=args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    if model is None:
        model = MLP(input_size=36 * (args.sampled_points + 1), num_layers=args.num_mlp_layers,
                    num_neurons_per_layer=args.num_neurons_per_layer, output_size=args.output_dim).to(device)

        # Load the saved state dictionary
        model_path = r"C:\Users\benjy\Downloads\best.pth"  # Update with the path to your saved model
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
            data, lpe, info, pe = batch['point_cloud'].to(device), batch['lpe'].to(device), batch['info'], batch[
                'pe'].to(device)
            if args.output_dim == 4:
                label = info['class'].to(device).long()
            if args.use_second_deg:
                x, y, z = data.unbind(dim=2)
                data = torch.stack([x ** 2, x * y, x * z, y ** 2, y * z, z ** 2, x, y, z], dim=2)
            if args.lpe_dim != 0:
                data = torch.cat([data, lpe], dim=2).to(device)
            if args.PE_dim != 0:
                data = torch.cat([data, pe], dim=2).to(device)
            if args.use_xyz == 0:
                data = data[:, :, 3:]
            data = data.permute(0, 2, 1)

            output = model(data)

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
    testPretrainedModel(args, model=model)
    torch.save(model.state_dict(), "best_2.pth")

