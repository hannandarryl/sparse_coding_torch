from sklearn.decomposition import PCA
import numpy as np
import torch

class PCASparseActivations:
    def __init__(self, data_loader, sparse_layer, batch_size, device):
        super().__init__()
        
        self.device = device
        
        example_data = next(iter(data_loader))
        
        u_init = torch.zeros([batch_size, sparse_layer.out_channels] +
                    sparse_layer.get_output_shape(example_data[1]))
        
        print('Fitting pca...')
        X = []
        for labels, local_batch in data_loader:
                if u_init.size(0) != local_batch.size(0):
                    u_init = torch.zeros([local_batch.size(0), sparse_layer.out_channels] +
                        sparse_layer.get_output_shape(example_data[1]))

                local_batch = local_batch.to(device)

                activations, u_init = sparse_layer(local_batch, u_init)
                
                for a in activations:
                    X.append(a.flatten().detach().cpu().numpy())

        X = np.array(X)

        self.pca = PCA(n_components=40)
        self.pca.fit(X)
        print('PCA complete!')
        
    def get_components(self, activations):
        comps = []
        for a in activations:
            comps.append(self.pca.transform(a.flatten().detach().cpu().numpy().reshape(1, -1)))
        
        return torch.tensor(np.stack(comps), device=self.device)