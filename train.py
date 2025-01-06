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
from torch.profiler import profile, record_function, ProfilerActivity
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
            output = model((pcl.permute(0, 2, 1)).unsqueeze(2))
            output = output.squeeze()
            loss = loss_function(output, label)
            output = output[:,:args.output_dim]
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
        wandb.init(project=args.wandb_proj, name=args.exp_name)

    print(device)
    print(args)
    num_epochs = args.epochs
    learning_rate = args.lr
    if args.output_dim==5:
        train_dataset = BasicPointCloudDataset(file_path="train_surfaces_05X05.h5" , args=args)
        test_dataset = BasicPointCloudDataset(file_path='test_surfaces_05X05.h5' , args=args)
    if args.output_dim==4:
        train_dataset = BasicPointCloudDataset(file_path="train_surfaces_05X05_no_edge.h5" , args=args)
        test_dataset = BasicPointCloudDataset(file_path="test_surfaces_05X05_no_edge.h5" , args=args)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = shapeClassifier(args).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Num of parameters in NN: {num_params}')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # milestones = np.linspace(args.lr_jumps,num_epochs,num_epochs//args.lr_jumps)
    milestones = [args.lr_jumps * (i) for i in range(1,num_epochs//args.lr_jumps + 1)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    tripletMarginLoss = nn.TripletMarginLoss(margin=args.contr_margin)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    mseLoss = nn.MSELoss()
    contr_loss_weight = args.contr_loss_weight
    data_in_1_cube = args.cube
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
                if data_in_1_cube:
                    pcl /= torch.max(torch.abs(pcl))
                label = info['class'].to(device).long()
                output = (model((pcl.permute(0, 2, 1)).unsqueeze(2))).squeeze()

                orig_classification = output[:, :args.output_dim]
                orig_emb = output
                classification_loss = torch.tensor((0))
                if args.classification == 1:
                    classification_loss = criterion(orig_classification, label)

                if args.contr_loss_weight != 0:
                    pcl2 = batch['point_cloud2'].to(device)
                    contrastive_point_cloud = batch['contrastive_point_cloud'].to(device)

                    output_pcl2 = (model((pcl2.permute(0, 2, 1)).unsqueeze(2))).squeeze()
                    pos_emb = output_pcl2
                    output_contrastive_pcl = (model((contrastive_point_cloud.permute(0, 2, 1)).unsqueeze(2))).squeeze()
                    neg_emb = output_contrastive_pcl
                    contrstive_loss = tripletMarginLoss(orig_emb, pos_emb, neg_emb)
                    total_train_contrastive_positive_loss += mseLoss(orig_emb, pos_emb)
                    total_train_contrastive_negative_loss += mseLoss(orig_emb, neg_emb)
                    total_train_contrastive_loss += contrstive_loss
                else:
                    contrstive_loss = torch.tensor((0))


                new_awesome_loss = classification_loss + (contr_loss_weight * contrstive_loss)
                optimizer.zero_grad()
                new_awesome_loss.backward()
                optimizer.step()

                current_lr = optimizer.param_groups[0]['lr']

                total_train_loss += classification_loss.item()
                preds = orig_classification.max(dim=1)[1]
                total_train_acc_loss += torch.mean((preds == label).float()).item()

                count = count + 1

                tqdm_bar.set_postfix(train_loss=f'{(classification_loss.item()):.4f}')

        classification_train_loss = (total_train_loss / count)
        classification_acc_train = (total_train_acc_loss / (count))
        contrastive_train_loss = (total_train_contrastive_loss / (count))
        contrastive_positive_train_loss = (total_train_contrastive_positive_loss / (count))
        contrastive_negative_train_loss = (total_train_contrastive_negative_loss / (count))
        print(f'contrastive_loss: {contrastive_train_loss}')
        print(f'contrastive_positive_MSEloss: {contrastive_positive_train_loss}')
        print(f'contrastive_negative_MSEloss: {contrastive_negative_train_loss}')

        if args.classification == 1:
            test_loss, acc_test, label_accuracies = test(model, test_dataloader, criterion, device, args)
            print(f'LR: {current_lr}')

            print({"epoch": epoch, "train_loss": classification_train_loss ,"test_loss": test_loss, "acc_train": classification_acc_train, "acc_test": acc_test})
            for key in label_accuracies:
                print("label_" + str(key), ":", label_accuracies[key])

        scheduler.step()

        if args.use_wandb:
            if args.classification == 1:
                wandb.log({"epoch": epoch, "train_loss": classification_train_loss ,"test_loss": test_loss, "acc_train": classification_acc_train, "acc_test": acc_test})
                for key in label_accuracies:
                    wandb.log({"epoch": epoch, "label_"+str(key) : label_accuracies[key]})
            if args.contr_loss_weight!=0:
                wandb.log({"epoch": epoch, "contrastive_loss":contrastive_train_loss})
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
    parser.add_argument('--use_wandb', type=int, default=0, metavar='N',
                        help='use angles in learning ')
    parser.add_argument('--contr_margin', type=float, default=5.0, metavar='N',
                        help='margin used for contrastive loss')
    parser.add_argument('--use_lap_reorder', type=int, default=1, metavar='N',
                        help='reorder points by laplacian order ')
    parser.add_argument('--lap_eigenvalues_dim', type=int, default=15, metavar='N',
                        help='use eigenvalues in as input')
    parser.add_argument('--use_second_deg', type=int, default=1, metavar='N',
                        help='use second degree embedding ')
    parser.add_argument('--lpe_normalize', type=int, default=1, metavar='N',
                        help='use normalized laplacian')
    parser.add_argument('--std_dev', type=float, default=0.01, metavar='N',
                        help='amount of noise to add to data')
    parser.add_argument('--max_curve_diff', type=float, default=0.2, metavar='N',
                        help='max difference in curvature for contrastive loss')
    parser.add_argument('--min_curve_diff', type=float, default=0.05, metavar='N',
                        help='min difference in curvature for contrastive loss')
    parser.add_argument('--clip', type=float, default=0.05, metavar='N',
                        help='clip noise')
    parser.add_argument('--contr_loss_weight', type=float, default=0.0, metavar='N',
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
    label_correct = {label: 0 for label in range(args.output_dim)}
    label_total = {label: 0 for label in range(args.output_dim)}
    wrong_preds = {label: [] for label in range(args.output_dim)}
    wrong_idx = {label: [] for label in range(args.output_dim)}
    wrong_pcl = {label: [] for label in range(args.output_dim)}
    wrong_pred_class = {label: [] for label in range(args.output_dim)}
    wrong_K1_values = {label: [] for label in range(args.output_dim)}
    wrong_K2_values = {label: [] for label in range(args.output_dim)}

    with torch.no_grad():
        for batch in test_dataloader:
            pcl, info = batch['point_cloud'].to(device), batch['info']
            label = info['class'].to(device).long()
            output = model((pcl.permute(0, 2, 1)).unsqueeze(2))
            output = (output[:,:args.output_dim]).squeeze()
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
            for label_name in range(args.output_dim):
                correct_mask = (preds == label_name) & (label == label_name)
                label_correct[label_name] += correct_mask.sum().item()
                label_total[label_name] += (label == label_name).sum().item()

    label_accuracies = {
        label: label_correct[label] / label_total[label]
        for label in range(args.output_dim)
        if label_total[label] != 0
    }
    for label, accuracy in label_accuracies.items():
        print(f"Accuracy for label {label}: {accuracy:.4f}")

    # for label in range(4):
    for label in range(args.output_dim):
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
    # cProfile.runctx('train_and_test(args)')
    # args.epochs=1
    # profiler = cProfile.Profile()
    # cProfile.runctx('train_and_test(args=args)', globals(), locals(), sort='tottime')
    # stats = pstats.Stats(profiler)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    # exit(0)
    model = train_and_test(args)
    torch.save(model.state_dict(), f'{args.exp_name}.pt')
    # model = input_visualized_importance()
    testPretrainedModel(args, model=model.to('cuda'))


