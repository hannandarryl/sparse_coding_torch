import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
from load_data import load_bamc_data
from conv_sparse_model import ConvSparseLayer
import time
import numpy as np
    
class SmallDataClassifier(nn.Module):
    
    def __init__(self, sparse_layer):
        super().__init__()

        self.sparse_layer = sparse_layer

        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
        
#         self.compress_activations = nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1, 10, 10), stride=(1, 4, 4), padding=(0, 0, 0))
        
#         self.compress_time = nn.Conv3d(in_channels=27, out_channels=4, kernel_size=1, stride=1, padding=0)
        
        # First fully connected layer
        self.fc1 = nn.Linear(1022112, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    # x represents our data
    def forward(self, x, u_init):
        # Pass data through conv1
        activations, u = self.sparse_layer(x, u_init)
        
#         x = self.compress_activations(activations).permute(0, 2, 1, 3, 4)
#         x = self.compress_time(activations.permute(0, 2, 1, 3, 4))
        
        # x = self.dropout3d(x)
        
        # Flatten x with start_dim=1
        x = torch.flatten(activations, 1)
        
        # print(x.shape)
        
        # Pass data through fc1
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

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
        
        best_so_far = float('inf')

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

        for param in frozen_sparse.parameters():
            param.requires_grad = False

        predictive_model = torch.nn.DataParallel(SmallDataClassifier(frozen_sparse), device_ids=[0,1,2,3])
        predictive_model.to(device)

        prediction_optimizer = torch.optim.Adam(predictive_model.parameters(),
                                                lr=args.lr)

        criterion = torch.nn.BCEWithLogitsLoss()
        
        predictive_model.train()
        for epoch in range(args.epochs):
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
                torch_labels[[i for i in range(len(labels)) if labels[i] == 'No_Sliding']] = 1
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
                error = None

                u_init = torch.zeros([batch_size, frozen_sparse.out_channels] +
                            frozen_sparse.get_output_shape(example_data[1]))

                # for local_batch in train_loader:
                for labels, local_batch in test_loader:
                    if u_init.size(0) != local_batch.size(0):
                        u_init = torch.zeros([local_batch.size(0), frozen_sparse.out_channels] +
                            frozen_sparse.get_output_shape(example_data[1]))

                    local_batch = local_batch.to(device)

                    torch_labels = torch.zeros(len(labels))
                    torch_labels[[i for i in range(len(labels)) if labels[i] == 'No_Sliding']] = 1
                    torch_labels = torch_labels.unsqueeze(1).to(device)


                    pred, _, u_init = predictive_model(local_batch, u_init)

                    if error is None:
                        error = torch.abs(torch_labels - torch.nn.Sigmoid()(pred).round()).flatten()
                    else:
                        error = torch.cat((error, torch.abs(torch_labels - torch.nn.Sigmoid()(pred).round()).flatten()))

                t2 = time.perf_counter()
                
                mean_error = error.mean().cpu().numpy()

                print('fold={}, epoch={}, time={:.2f}, loss={:.2f}, error={:.2f}'.format(i_fold, epoch, t2-t1, epoch_loss, mean_error))
            
            if mean_error < best_so_far:
                print("found better model")
                # Save model parameters
                torch.save({
                    'model_state_dict': predictive_model.module.state_dict(),
                    'optimizer_state_dict': prediction_optimizer.state_dict(),
                }, os.path.join(output_dir, "model-best.pt"))
                best_so_far = mean_error
                
        checkpoint = torch.load(os.path.join(output_dir, "model-best.pt"))
        predictive_model.module.load_state_dict(checkpoint['model_state_dict'])

        predictive_model.eval()
        with torch.no_grad():
            epoch_loss = 0

            y_h = None
            y = None

            error = None


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
                torch_labels[[i for i in range(len(labels)) if labels[i] == 'No_Sliding']] = 1
                torch_labels = torch_labels.unsqueeze(1).to(device)


                pred, _, u_init = predictive_model(local_batch, u_init)

                loss = criterion(pred, torch_labels)
                epoch_loss += loss.item() * local_batch.size(0)

                if error is None:
                    error = torch.abs(torch_labels - torch.nn.Sigmoid()(pred).round()).flatten()
                    y_h = torch.nn.Sigmoid()(pred).round().flatten()
                    y = torch_labels.flatten()
                else:
                    error = torch.cat((error, torch.abs(torch_labels - torch.nn.Sigmoid()(pred).round()).flatten()))
                    y_h = torch.cat((y_h, torch.nn.Sigmoid()(pred).round().flatten()))
                    y = torch.cat((y, torch_labels.flatten()))

            t2 = time.perf_counter()

            print('fold={}, loss={:.2f}, time={:.2f}'.format(i_fold, loss, t2-t1))
            
            mean_error = error.mean().cpu().numpy()
            all_errors.append(mean_error)

            print("Test error={:.2f}, fold={}".format(mean_error, i_fold))
            
    print("Final error={:.2f}".format(np.array(all_errors).mean()))