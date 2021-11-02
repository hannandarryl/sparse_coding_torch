import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
from utils.load_data import load_bamc_data, load_covid_data, load_bamc_clips
from feature_extraction.conv_model import ConvLayer
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
    
class SmallDataClassifier(nn.Module):
    
    def __init__(self, pretrained_cnn):
        super().__init__()

        self.pretrained_cnn = pretrained_cnn
        
        self.compress_activations = nn.Conv3d(in_channels=64, out_channels=24, kernel_size=(1, 8, 8), stride=(1, 4, 4), padding=(1, 2, 2))
        
#         self.compress_time = nn.Conv3d(in_channels=27, out_channels=4, kernel_size=1, stride=1, padding=0)
        
        # First fully connected layer
        self.fc1 = nn.Linear(641424, 1000)
    # #         self.fc1 = nn.Linear(73008, 128)
    #         self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 20)
        self.fc4 = nn.Linear(20, 1)

    # x represents our data
    def forward(self, x):
        # Pass data through conv1
        activations = F.relu(self.pretrained_cnn.get_activations(x))
        
        x = F.relu(self.compress_activations(activations))
#         x = self.compress_time(activations.permute(0, 2, 1, 3, 4))
        
        # x = self.dropout3d(x)
        
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--kernel_height', default=16, type=int)
    parser.add_argument('--kernel_width', default=16, type=int)
    parser.add_argument('--kernel_depth', default=4, type=int)
    parser.add_argument('--num_kernels', default=64, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--output_dir', default='./output', type=str)
#     parser.add_argument('--checkpoint', required=True, type=str)
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
    
    splits, dataset = load_bamc_clips(batch_size, 0.8, sparse_model=None, device=None, num_frames=args.kernel_depth, seed=42)
    
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

        
        best_so_far = float('inf')

        frozen = ConvLayer(in_channels=1,
                               out_channels=args.num_kernels,
                               kernel_size=(args.kernel_depth, args.kernel_height, args.kernel_width),
                               stride=args.stride,
                               padding=0,
                               convo_dim=3)

        predictive_model = torch.nn.DataParallel(SmallDataClassifier(frozen))
        predictive_model.to(device)

        prediction_optimizer = torch.optim.Adam(predictive_model.parameters(),
                                                lr=args.lr)
        
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(args.epochs):
            predictive_model.train()
            epoch_loss = 0
            # for local_batch in train_loader:
            t1 = time.perf_counter()

            for labels, local_batch, vid_f in tqdm(train_loader):
                local_batch = local_batch.to(device)

                torch_labels = torch.zeros(len(labels))
                torch_labels[[i for i in range(len(labels)) if labels[i] == 'PTX_No_Sliding']] = 1
                torch_labels = torch_labels.unsqueeze(1).to(device)
   
                pred = predictive_model(local_batch)

                loss = criterion(pred, torch_labels)
                # loss += frozen_sparse.loss(local_batch, activations)
                epoch_loss += loss.item() * local_batch.size(0)

                prediction_optimizer.zero_grad()
                loss.backward()
                prediction_optimizer.step()
                
            t2 = time.perf_counter()

            print('fold={}, epoch={}, time={:.2f}, loss={:.2f}'.format(i_fold, epoch, t2-t1, epoch_loss))
            
            if epoch_loss <= best_so_far:
                print("found better model")
                # Save model parameters
                torch.save({
                    'model_state_dict': predictive_model.module.state_dict(),
                    'optimizer_state_dict': prediction_optimizer.state_dict(),
                }, os.path.join(output_dir, "model-best.pt"))
                best_so_far = epoch_loss
                
        checkpoint = torch.load(os.path.join(output_dir, "model-best.pt"))
        predictive_model.module.load_state_dict(checkpoint['model_state_dict'])

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


                pred = predictive_model(local_batch)#, u_init)

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
            
            i_fold += 1
            
    print("Final error={:.2f}".format(np.array(all_errors).mean()))