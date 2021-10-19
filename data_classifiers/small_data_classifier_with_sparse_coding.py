import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
from utils.load_data import load_bamc_data
from feature_extraction.conv_sparse_model import ConvSparseLayer
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from data_classifiers.pca import PCASparseActivations
    
class SmallDataClassifier(nn.Module):
    
    def __init__(self, sparse_layer, pca=None):
        super().__init__()

        self.sparse_layer = sparse_layer
        self.pca = pca

#         self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
        if self.pca:
            self.fc1 = nn.Linear(40, 20)
            self.fc2 = nn.Linear(20, 10)
            self.fc3 = nn.Linear(10, 1)
        else:
            self.compress_activations_conv = nn.Conv3d(in_channels=48, out_channels=24, kernel_size=(1, 8, 8), stride=(1, 2, 2), padding=(0, 0, 0))
    #         self.compress_activations_ff = nn.Linear(18*18, 100)

    #         self.compress_time = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(14, 1, 1), stride=(1, 1, 1), padding=0)
            self.compress_time = nn.MaxPool3d(kernel_size=(14, 1, 1), stride=(1, 1, 1), padding=0)
    #         self.compress_time = nn.GRU(input_size=14, hidden_size=1)
    #         self.compress_time = nn.MaxPool2d(kernel_size=(14, 1), stride=1, padding=0)
    #         self.compress_time = nn.Linear(14, 10)

    #         self.compress_features = nn.Linear(48, 1)

            self.feature_weights = nn.Parameter(torch.rand(48),
                                        requires_grad=True)

            # First fully connected layer
            self.fc1 = nn.Linear(6144, 1000)
    # #         self.fc1 = nn.Linear(73008, 128)
    #         self.relu = nn.ReLU()
            self.fc2 = nn.Linear(1000, 100)
            self.fc3 = nn.Linear(100, 20)
            self.fc4 = nn.Linear(20, 1)

    # x represents our data
    def forward(self, x, u_init):
        # Pass data through conv1
        activations, u = self.sparse_layer(x, u_init)
        
        if self.pca:
            x = pca.get_components(activations)
            x = F.relu(self.fc1(x)).squeeze(-1)
            x = F.relu(self.fc2(x)).squeeze(-1)
            x = self.fc3(x).squeeze(-1)
            return x, activations, u
        else:
        
            batch_size, channel_size, time_size, width_size, height_size = activations.size()

            x = torch.zeros_like(activations)

            for i in range(channel_size):
                x[:, i, :, :, :] = activations[:, i, :, :, :] * self.feature_weights[i]

            x = F.relu(self.compress_activations_conv(x))#.view(batch_size, channel_size, time_size, -1)
    #         x = F.relu(self.compress_activations_ff(x)).view(batch_size, channel_size, -1, time_size)

            x = F.relu(self.compress_time(x))

    #         x = torch.flatten(x, 1)

    #         x = self.fc1(x)

    #         x = self.compress_features(x)
    #         x, _ = self.compress_time(x)
    #         x = x.squeeze(2)

            x = torch.flatten(x, 1)

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
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

        return x, activations, u

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--kernel_height', default=16, type=int)
    parser.add_argument('--kernel_width', default=16, type=int)
    parser.add_argument('--kernel_depth', default=8, type=int)
    parser.add_argument('--num_kernels', default=48, type=int)
    parser.add_argument('--stride', default=4, type=int)
    parser.add_argument('--max_activation_iter', default=1000, type=int)
    parser.add_argument('--activation_lr', default=1e-4, type=float)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lam', default=0.05, type=float)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--num_folds', default=1, type=int)
    parser.add_argument('--pca', action='store_true')
    
    args = parser.parse_args()
    
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
    
    for i_fold in range(args.num_folds):
        train_loader, test_loader = load_bamc_data(batch_size, 0.8)
        print('Loaded', len(train_loader), 'train examples')
        print('Loaded', len(test_loader), 'test examples')

        example_data = next(iter(train_loader))
        
        best_so_far = float('-inf')

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
        sparse_param = torch.load(args.checkpoint)
        frozen_sparse.load_state_dict(sparse_param['model_state_dict'])
        frozen_sparse.to(device)

        for param in frozen_sparse.parameters():
            param.requires_grad = False
            
        pca = None
        
        if args.pca:
            pca = PCASparseActivations(train_loader, frozen_sparse, batch_size, device) 

        predictive_model = SmallDataClassifier(frozen_sparse, pca)
#         predictive_model = torch.nn.DataParallel(SmallDataClassifier(frozen_sparse, pca), device_ids=[0,1,2,3])
        predictive_model.to(device)

        prediction_optimizer = torch.optim.Adam(predictive_model.parameters(),
                                                lr=args.lr)

        criterion = torch.nn.BCEWithLogitsLoss()
        
        for epoch in range(args.epochs):
            predictive_model.train()
            epoch_loss = 0
            # for local_batch in train_loader:
            t1 = time.perf_counter()
            u_init = torch.zeros([batch_size, frozen_sparse.out_channels] +
                        frozen_sparse.get_output_shape(example_data[1]))

            for labels, local_batch in tqdm(train_loader):
                if u_init.size(0) != local_batch.size(0):
                    u_init = torch.zeros([local_batch.size(0), frozen_sparse.out_channels] +
                        frozen_sparse.get_output_shape(example_data[1]))

                local_batch = local_batch.to(device)

                torch_labels = torch.zeros(len(labels))
                torch_labels[[i for i in range(len(labels)) if labels[i] == 'PTX_No_Sliding']] = 1
                torch_labels = torch_labels.unsqueeze(1).to(device)

                pred, activations, u_init = predictive_model(local_batch, u_init)

                loss = criterion(pred, torch_labels)
                # loss += frozen_sparse.loss(local_batch, activations)
                epoch_loss += loss.item() * local_batch.size(0)

                prediction_optimizer.zero_grad()
                loss.backward()
                prediction_optimizer.step()

            t2 = time.perf_counter()
            
            predictive_model.eval()
            with torch.no_grad():
                y_true = None
                y_pred = None

                u_init = torch.zeros([batch_size, frozen_sparse.out_channels] +
                            frozen_sparse.get_output_shape(example_data[1]))

                # for local_batch in train_loader:
                for labels, local_batch in test_loader:
                    if u_init.size(0) != local_batch.size(0):
                        u_init = torch.zeros([local_batch.size(0), frozen_sparse.out_channels] +
                            frozen_sparse.get_output_shape(example_data[1]))

                    local_batch = local_batch.to(device)

                    torch_labels = torch.zeros(len(labels))
                    torch_labels[[i for i in range(len(labels)) if labels[i] == 'PTX_No_Sliding']] = 1
                    torch_labels = torch_labels.unsqueeze(1).to(device)


                    pred, _, u_init = predictive_model(local_batch, u_init)

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
            
            if accuracy > best_so_far:
                print("found better model")
                # Save model parameters
                torch.save({
                    'model_state_dict': predictive_model.state_dict(),
                    'optimizer_state_dict': prediction_optimizer.state_dict(),
                }, os.path.join(output_dir, "model-best.pt"))
                best_so_far = f1
                
        checkpoint = torch.load(os.path.join(output_dir, "model-best.pt"))
        predictive_model.load_state_dict(checkpoint['model_state_dict'])

        predictive_model.eval()
        with torch.no_grad():
            epoch_loss = 0

            y_true = None
            y_pred = None


            u_init = torch.zeros([batch_size, frozen_sparse.out_channels] +
                        frozen_sparse.get_output_shape(example_data[1]))

            t1 = time.perf_counter()
            # for local_batch in train_loader:
            for labels, local_batch in test_loader:
                if u_init.size(0) != local_batch.size(0):
                    u_init = torch.zeros([local_batch.size(0), frozen_sparse.out_channels] +
                        frozen_sparse.get_output_shape(example_data[1]))

                local_batch = local_batch.to(device)

                torch_labels = torch.zeros(len(labels))
                torch_labels[[i for i in range(len(labels)) if labels[i] == 'PTX_No_Sliding']] = 1
                torch_labels = torch_labels.unsqueeze(1).to(device)


                pred, _, u_init = predictive_model(local_batch, u_init)

                loss = criterion(pred, torch_labels)
                epoch_loss += loss.item() * local_batch.size(0)

                if y_true is None:
                    y_true = torch_labels.detach().cpu().flatten().to(torch.long)
                    y_pred = torch.nn.Sigmoid()(pred).round().detach().cpu().flatten().to(torch.long)
                else:
                    y_true = torch.cat((y_true, torch_labels.detach().cpu().flatten().to(torch.long)))
                    y_pred = torch.cat((y_pred, torch.nn.Sigmoid()(pred).detach().round().cpu().flatten().to(torch.long)))

            t2 = time.perf_counter()

            print('fold={}, loss={:.2f}, time={:.2f}'.format(i_fold, loss, t2-t1))
            
            f1 = f1_score(y_true, y_pred, average='macro')
            accuracy = accuracy_score(y_true, y_pred)
            all_errors.append(accuracy)

            print("Test f1={:.2f}, acc={:.2f}, fold={}".format(f1, accuracy, i_fold))
            
    print("Final accuracy={:.2f}".format(np.array(f1).mean()))