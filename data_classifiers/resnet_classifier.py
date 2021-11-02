import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
from utils.load_data import load_bamc_data, load_covid_data
import time
import numpy as np
from torchvision.models import resnet101
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    
class SmallDataClassifier(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.resnet = resnet101(pretrained=True)
        
        self.ff = nn.Sequential(nn.Linear(2048, 128), nn.ReLU())
        self.resnet.fc = self.ff
        
        self.compress_time = nn.GRU(input_size=128, hidden_size=128)
        
        self.out = nn.Linear(128, 1)

    # x represents our data
    def forward(self, x):
        batch_size, channel_size, time_size, width, height = x.size()
        x = x.view(batch_size * time_size, channel_size, width, height)
        
        # Pass data through conv1
        x = self.resnet(x)
        
        x = x.view(time_size, batch_size, -1)
        
        x, _ = self.compress_time(x)
        x = x.squeeze(2)
        x = x[-1]
            
        x = self.out(x)

        return x
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--output_dir', default='./output', type=str)
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
    
    splits, dataset = load_bamc_data(batch_size, 0.8, seed=42)
    
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
        
        model = SmallDataClassifier()

        predictive_model = torch.nn.DataParallel(model)
        predictive_model.to(device)
        
        for param in predictive_model.module.resnet.parameters():
            param.requires_grad = False
            
        for param in predictive_model.module.resnet.fc.parameters():
            param.requires_grad = True

        prediction_optimizer = torch.optim.Adam(predictive_model.parameters(),
                                                lr=args.lr)

        criterion = torch.nn.BCEWithLogitsLoss()
        
        for epoch in range(args.epochs):
            predictive_model.train()
            epoch_loss = 0
            # for local_batch in train_loader:
            t1 = time.perf_counter()

            for labels, local_batch in tqdm(train_loader):
                local_batch = local_batch.to(device).squeeze(2)

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
            
            predictive_model.eval()
            with torch.no_grad():
                
                y_true = None
                y_pred = None

                # for local_batch in train_loader:
                for labels, local_batch in test_loader:
                    local_batch = local_batch.to(device).squeeze(2)

                    torch_labels = torch.zeros(len(labels))
                    torch_labels[[i for i in range(len(labels)) if labels[i] == 'PTX_No_Sliding']] = 1
                    torch_labels = torch_labels.unsqueeze(1).to(device)

                    pred = predictive_model(local_batch)

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
            
            if accuracy >= best_so_far:
                print("found better model")
                # Save model parameters
                torch.save({
                    'model_state_dict': predictive_model.module.state_dict(),
                    'optimizer_state_dict': prediction_optimizer.state_dict(),
                }, os.path.join(output_dir, "model-best.pt"))
                best_so_far = accuracy
                
        checkpoint = torch.load(os.path.join(output_dir, "model-best.pt"))
        predictive_model.module.load_state_dict(checkpoint['model_state_dict'])

        predictive_model.eval()
        with torch.no_grad():
            epoch_loss = 0

            y_h = None
            y = None

            error = None

            t1 = time.perf_counter()
            # for local_batch in train_loader:
            for labels, local_batch in test_loader:

                local_batch = local_batch.to(device).squeeze(2)

                torch_labels = torch.zeros(len(labels))
                torch_labels[[i for i in range(len(labels)) if labels[i] == 'PTX_No_Sliding']] = 1
                torch_labels = torch_labels.unsqueeze(1).to(device)

                pred = predictive_model(local_batch)

                loss = criterion(pred, torch_labels)
                epoch_loss += loss.item() * local_batch.size(0)
                
                if y_true is None:
                    y_true = torch_labels.detach().cpu().flatten().to(torch.long)
                    y_pred = torch.nn.Sigmoid()(pred).round().detach().cpu().flatten().to(torch.long)
                else:
                    y_true = torch.cat((y_true, torch_labels.detach().cpu().flatten().to(torch.long)))
                    y_pred = torch.cat((y_pred, torch.nn.Sigmoid()(pred).round().detach().cpu().flatten().to(torch.long)))

                t2 = time.perf_counter()
                
                f1 = f1_score(y_true, y_pred, average='macro')
                accuracy = accuracy_score(y_true, y_pred)

            print("Test f1={:.2f}, acc={:.2f}, fold={}".format(f1, accuracy, i_fold))
            
            i_fold += 1
            
    print("Final error={:.2f}".format(np.array(all_errors).mean()))