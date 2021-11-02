import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
from utils.load_data import load_bamc_data, load_bamc_clips
from feature_extraction.conv_sparse_model import ConvSparseLayer
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import random
from data_classifiers.pca import PCASparseActivations
    
class SmallDataClassifier(nn.Module):
    
    def __init__(self, sparse_layer, pca=None):
        super().__init__()

#         self.sparse_layer = sparse_layer
        self.pca = pca

#         self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
        if self.pca:
            self.fc1 = nn.Linear(40, 20)
            self.fc2 = nn.Linear(20, 10)
            self.fc3 = nn.Linear(10, 1)
        else:
            self.compress_activations_conv_1 = nn.Conv3d(in_channels=100, out_channels=64, kernel_size=(8, 8, 8), stride=(4, 4, 4), padding=(2, 4, 4))
#             self.max_pool_1 = nn.MaxPool3d(kernel_size=(1, 4, 4))
            self.compress_activations_conv_2 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(1, 8, 8), stride=(1, 2, 2), padding=(1, 4, 4))
#             self.max_pool_2 = nn.MaxPool3d(kernel_size=(1, 4, 4))
            self.compress_activations_conv_3 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(1, 2, 2))
#             self.max_pool_3 = nn.MaxPool3d(kernel_size=(1, 4, 4))
            self.compress_activations_ff = nn.Linear(4576, 100)

    #         self.compress_time = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(14, 1, 1), stride=(1, 1, 1), padding=0)
#             self.compress_time = nn.MaxPool3d(kernel_size=(14, 1, 1), stride=(1, 1, 1), padding=0)
            self.compress_time = nn.GRU(input_size=100, hidden_size=100)
    #         self.compress_time = nn.MaxPool2d(kernel_size=(14, 1), stride=1, padding=0)
    #         self.compress_time = nn.Linear(14, 10)

#             self.compress_features = nn.Linear(48, 10)

#             self.feature_weights = nn.Parameter(torch.rand(48),
#                                         requires_grad=True)

            # First fully connected layer
#             self.fc1 = nn.Linear(729, 100)
    # #         self.fc1 = nn.Linear(73008, 128)
    #         self.relu = nn.ReLU()
#             self.fc2 = nn.Linear(1000, 100)
            self.fc3 = nn.Linear(100, 20)
            self.fc4 = nn.Linear(20, 1)

    # x represents our data
    def forward(self, activations):
        # Pass data through conv1
#         activations, u = self.sparse_layer(x, u_init)
        
        if self.pca:
            x = pca.get_components(activations)
            x = F.relu(self.fc1(x)).squeeze(-1)
            x = F.relu(self.fc2(x)).squeeze(-1)
            x = self.fc3(x).squeeze(-1)
            return x, activations, u
        else:
        
            batch_size, time_size, channel_size, height_size, width_size = activations.size()
            
            activations = activations.view(batch_size, channel_size, time_size, height_size, width_size)

#             x = torch.zeros_like(activations)

#             for i in range(channel_size):
#                 x[:, i, :, :, :] = activations[:, i, :, :, :] * self.feature_weights[i]

            x = F.relu(self.compress_activations_conv_1(activations))#.view(time_size, batch_size, -1)
            x = F.relu(self.compress_activations_conv_2(x))
            x = F.relu(self.compress_activations_conv_3(x)).view(batch_size, 19, -1)
            x = F.relu(self.compress_activations_ff(x)).permute(1, 0, 2)

#             x = F.relu(self.compress_time(x))

    #         x = torch.flatten(x, 1)

    #         x = self.fc1(x)

#             x = F.relu(self.compress_features(x))
            x, _ = self.compress_time(x)
            x = x[-1]
            x = x.view(batch_size, -1)

#             x = F.relu(self.fc1(x))
#             x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)

            # x = self.dropout3d(x)

            # Flatten x with start_dim=1
    #         x = torch.flatten(x, 1)

            # print(x.shape)

    #         activations = activations.view(batch_size, time_size, -1)

            # Pass data through fc1
    #         x = self.fc1(x)
    #         x = self.relu(x)
    # #         x = self.dropout(x)
    #         x = self.fc2(x).permute(0, 2, 1)

    #         x = self.compress_time(x)

    #         x = x.squeeze(-1)

        return x, activations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--kernel_height', default=16, type=int)
    parser.add_argument('--kernel_width', default=16, type=int)
    parser.add_argument('--kernel_depth', default=1, type=int)
    parser.add_argument('--num_kernels', default=100, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--max_activation_iter', default=75, type=int)
    parser.add_argument('--activation_lr', default=1e-2, type=float)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lam', default=0.01, type=float)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--sparse_checkpoint', default=None, type=str)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--num_folds', default=1, type=int)
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--train_sparse', action='store_true')
    parser.add_argument('--recon_scale', default=1.0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--use_clips', action='store_true')
    parser.add_argument('--train', action='store_true')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(os.path.join(output_dir, 'arguments.txt'), 'w+') as out_f:
        out_f.write(str(args))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        batch_size = 1
    else:
        batch_size = args.batch_size

    all_errors = []
    
    frozen_sparse = ConvSparseLayer(in_channels=1,
                                   out_channels=args.num_kernels,
                                   kernel_size=(args.kernel_depth, args.kernel_height, args.kernel_width),
                                   stride=args.stride,
                                   padding=(0, 0, 0),
                                   convo_dim=3,
                                   rectifier=True,
                                   lam=args.lam,
                                   max_activation_iter=args.max_activation_iter,
                                   activation_lr=args.activation_lr)
    if args.sparse_checkpoint:
        sparse_param = torch.load(args.sparse_checkpoint)
        sparse_param['model_state_dict']['filters'] = sparse_param['model_state_dict']['module.filters']
        del sparse_param['model_state_dict']['module.filters']
        frozen_sparse.load_state_dict(sparse_param['model_state_dict'])

#         frozen_sparse.import_opencv_dir('/home/dwh48@drexel.edu/sparse_coding_torch/eds_weights')

    frozen_sparse.to(device)
    
#     splits, dataset = load_bamc_clips(batch_size, None, sparse_model=frozen_sparse, device=device, num_frames=args.kernel_depth, seed=args.seed)
    splits, dataset = load_bamc_data(batch_size, 0.8, sparse_model=frozen_sparse, device=device, seed=args.seed)
    
    overall_true = []
    overall_pred = []
    
    i_fold = 0
    for train_idx, test_idx in splits:
        
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   # shuffle=True,
                                                   sampler=train_sampler)

        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                        # shuffle=True,
                                                        sampler=test_sampler)

        
        best_so_far = float('-inf')
        
#         if args.use_clips:
#             splits = load_bamc_clips(batch_size, 0.8, sparse_model=frozen_sparse, device=device, num_frames=args.kernel_depth, seed=args.seed + i_fold)
#         else:
#             train_loader, test_loader = load_bamc_data(batch_size, 0.8, args.seed + i_fold)
#         print('Loaded', len(train_loader), 'train examples')
#         print('Loaded', len(test_loader), 'test examples')

#         example_data = next(iter(train_loader))

#         if not args.train_sparse:
#             for param in frozen_sparse.parameters():
#                 param.requires_grad = False
            
        pca = None
        
        if args.pca:
            pca = PCASparseActivations(train_loader, frozen_sparse, batch_size, device) 

#         predictive_model = SmallDataClassifier(frozen_sparse, pca)
        predictive_model = torch.nn.DataParallel(SmallDataClassifier(frozen_sparse, pca))
        predictive_model.to(device)
        
        criterion = torch.nn.BCEWithLogitsLoss()
        
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint)
            predictive_model.load_state_dict(checkpoint['model_state_dict'])
        
        if args.train:
            prediction_optimizer = torch.optim.Adam(predictive_model.parameters(),
                                                    lr=args.lr)

            for epoch in range(args.epochs):
                predictive_model.train()
                epoch_loss = 0
                # for local_batch in train_loader:
                t1 = time.perf_counter()
#                 u_init = torch.zeros([batch_size, frozen_sparse.out_channels] +
#                             frozen_sparse.get_output_shape(example_data[1]))

                for labels, local_batch, vid_f in tqdm(train_loader):
#                     if u_init.size(0) != local_batch.size(0):
#                         u_init = torch.zeros([local_batch.size(0), frozen_sparse.out_channels] +
#                             frozen_sparse.get_output_shape(example_data[1]))

                    local_batch = local_batch.to(device)

                    torch_labels = torch.zeros(len(labels))
                    torch_labels[[i for i in range(len(labels)) if labels[i] == 'PTX_No_Sliding']] = 1
                    torch_labels = torch_labels.unsqueeze(1).to(device)

                    pred, activations = predictive_model(local_batch)#, u_init)

                    loss = criterion(pred, torch_labels)
                    if args.train_sparse:
                        loss += args.recon_scale * frozen_sparse.loss(local_batch, activations)
                    epoch_loss += loss.item() * local_batch.size(0)

                    prediction_optimizer.zero_grad()
                    loss.backward()
                    prediction_optimizer.step()

                t2 = time.perf_counter()

                predictive_model.eval()
                with torch.no_grad():
                    y_true = None
                    y_pred = None

#                     u_init = torch.zeros([batch_size, frozen_sparse.out_channels] +
#                                 frozen_sparse.get_output_shape(example_data[1]))

                    # for local_batch in train_loader:
                    for labels, local_batch, vid_f in test_loader:
#                         if u_init.size(0) != local_batch.size(0):
#                             u_init = torch.zeros([local_batch.size(0), frozen_sparse.out_channels] +
#                                 frozen_sparse.get_output_shape(example_data[1]))

                        local_batch = local_batch.to(device)

                        torch_labels = torch.zeros(len(labels))
                        torch_labels[[i for i in range(len(labels)) if labels[i] == 'PTX_No_Sliding']] = 1
                        torch_labels = torch_labels.unsqueeze(1).to(device)


                        pred, _ = predictive_model(local_batch)#, u_init)

                        if y_true is None:
                            y_true = torch_labels.detach().cpu().flatten().to(torch.long)
                            y_pred = torch.nn.Sigmoid()(pred).round().detach().cpu().flatten().to(torch.long)
                        else:
                            y_true = torch.cat((y_true, torch_labels.detach().cpu().flatten().to(torch.long)))
                            y_pred = torch.cat((y_pred, torch.nn.Sigmoid()(pred).round().detach().cpu().flatten().to(torch.long)))

                    t2 = time.perf_counter()

                    f1 = f1_score(y_true, y_pred, average='macro')
                    accuracy = accuracy_score(y_true, y_pred)

                    print('fold={}, epoch={}, time={:.2f}, loss={:.2f}, f1={:.2f}, acc={:.2f}'.format(i_fold, epoch, t2-t1, epoch_loss, f1, accuracy))
#                 print('fold={}, epoch={}, time={:.2f}, loss={:.2f}'.format(i_fold, epoch, t2-t1, epoch_loss))

                if accuracy >= best_so_far:
                    print("found better model")
                    # Save model parameters
                    torch.save({
                        'model_state_dict': predictive_model.state_dict(),
                        'optimizer_state_dict': prediction_optimizer.state_dict(),
                    }, os.path.join(output_dir, "model-best.pt"))
                    best_so_far = accuracy

            checkpoint = torch.load(os.path.join(output_dir, "model-best.pt"))
            predictive_model.load_state_dict(checkpoint['model_state_dict'])

        predictive_model.eval()
        with torch.no_grad():
            epoch_loss = 0

            y_true = None
            y_pred = None
            
            pred_dict = {}
            gt_dict = {}

#             u_init = torch.zeros([batch_size, frozen_sparse.out_channels] +
#                         frozen_sparse.get_output_shape(example_data[1]))

            t1 = time.perf_counter()
            # for local_batch in train_loader:
            for labels, local_batch, vid_f in test_loader:
#                 if u_init.size(0) != local_batch.size(0):
#                     u_init = torch.zeros([local_batch.size(0), frozen_sparse.out_channels] +
#                         frozen_sparse.get_output_shape(example_data[1]))

                local_batch = local_batch.to(device)

                torch_labels = torch.zeros(len(labels))
                torch_labels[[i for i in range(len(labels)) if labels[i] == 'PTX_No_Sliding']] = 1
                torch_labels = torch_labels.unsqueeze(1).to(device)


                pred, _ = predictive_model(local_batch)#, u_init)

                loss = criterion(pred, torch_labels)
                epoch_loss += loss.item() * local_batch.size(0)
                
                for i, v_f in enumerate(vid_f):
                    if v_f not in pred_dict:
                        pred_dict[v_f] = torch.nn.Sigmoid()(pred[i]).round().detach().cpu().flatten().to(torch.long)
                    else:
                        pred_dict[v_f] = torch.cat((pred_dict[v_f], torch.nn.Sigmoid()(pred[i]).detach().round().cpu().flatten().to(torch.long)))
                        
                    if v_f not in gt_dict:
                        gt_dict[v_f] = torch_labels[i].detach().cpu().flatten().to(torch.long)
                    else:
                        gt_dict[v_f] = torch.cat((gt_dict[v_f], torch_labels[i].detach().cpu().flatten().to(torch.long)))

                if y_true is None:
                    y_true = torch_labels.detach().cpu().flatten().to(torch.long)
                    y_pred = torch.nn.Sigmoid()(pred).round().detach().cpu().flatten().to(torch.long)
                else:
                    y_true = torch.cat((y_true, torch_labels.detach().cpu().flatten().to(torch.long)))
                    y_pred = torch.cat((y_pred, torch.nn.Sigmoid()(pred).detach().round().cpu().flatten().to(torch.long)))

            t2 = time.perf_counter()
            
            vid_acc = []
            for k in pred_dict.keys():
                overall_true.append(torch.mode(gt_dict[k])[0])
                overall_pred.append(torch.mode(pred_dict[k])[0])
                if torch.mode(pred_dict[k])[0] == torch.mode(gt_dict[k])[0]:
                    vid_acc.append(1)
                else:
                    vid_acc.append(0)
                    
            vid_acc = np.array(vid_acc)
            
            print('----------------------------------------------------------------------------')
            for k in pred_dict.keys():
                print(k)
                print('Predictions:')
                print(pred_dict[k])
                print('Ground Truth:')
                print(gt_dict[k])
                print('----------------------------------------------------------------------------')

            print('fold={}, loss={:.2f}, time={:.2f}'.format(i_fold, loss, t2-t1))
            
            f1 = f1_score(y_true, y_pred, average='macro')
            accuracy = accuracy_score(y_true, y_pred)
            all_errors.append(np.sum(vid_acc) / len(vid_acc))

            print("Test f1={:.2f}, clip_acc={:.2f}, vid_acc={:.2f} fold={}".format(f1, accuracy, np.sum(vid_acc) / len(vid_acc), i_fold))
            
            print(confusion_matrix(y_true, y_pred))
            
#             with torch.no_grad():
#                 torch.cuda.empty_cache()
            
        i_fold = i_fold + 1
        
    overall_true = np.array(overall_true)
    overall_pred = np.array(overall_pred)
            
    final_f1 = f1_score(overall_true, overall_pred, average='macro')
    final_acc = accuracy_score(overall_true, overall_pred)
    final_conf = confusion_matrix(overall_true, overall_pred)
            
    print("Final accuracy={:.2f}, f1={:.2f}".format(final_acc, final_f1))
    print(final_conf)