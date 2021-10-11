import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib import cm
from conv_sparse_model import ConvSparseLayer
import os
import time
from tqdm import tqdm

def load_mnist_data(batch_size):
    batch_size_train = batch_size
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('~/Downloads/mnist/', train=True,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])), batch_size=batch_size_train,
        shuffle=True)
    return train_loader


def plot_filters(filters):
    num_filters = filters.shape[0]
    ncol = int(np.sqrt(num_filters))
    nrow = int(np.sqrt(num_filters))

    fig, axes = plt.subplots(ncols=ncol, nrows=nrow,
                             constrained_layout=True)

    ims = {}
    for i in range(num_filters):
        r = i // ncol
        c = i % ncol
        ims[(r, c)] = axes[r, c].imshow(filters[i, 0, :, :], cmap=cm.Greys_r)

    return plt


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        batch_size = 8
    else:
        batch_size = 64
        
    output_dir = 'mnist_out'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_loader = load_mnist_data(batch_size)
    example_data, example_targets = next(iter(train_loader))
    example_data = example_data.to(device)

    sparse_layer = ConvSparseLayer(in_channels=1,
                                   out_channels=16,
                                   kernel_size=8,
                                   stride=1,
                                   padding=0,
                                   lam=0.05, 
                                   activation_lr=1e-4,
                                   max_activation_iter=1000
                                   )
    sparse_layer.to(device)

    learning_rate = 1e-3
    filter_optimizer = torch.optim.Adam(sparse_layer.parameters(),
                                       lr=learning_rate)
    
    loss_log = []
    best_so_far = float('inf')

    for epoch in range(10):
        epoch_loss = 0
        epoch_start = time.perf_counter()
        
        u_init = torch.zeros([batch_size, sparse_layer.out_channels] +
                    sparse_layer.get_output_shape(example_data))
        
        for local_batch, local_labels in tqdm(train_loader):
            local_batch = local_batch.to(device)
            
            if local_batch.size(0) != batch_size:
                u_init = torch.zeros([local_batch.size(0), sparse_layer.out_channels] +
                    sparse_layer.get_output_shape(example_data))
            
            activations, u_init = sparse_layer(local_batch[:, :, :, :], u_init)
            loss = sparse_layer.loss(local_batch[:, :, :, :], activations)
            
            epoch_loss += loss.item() * local_batch.size(0)

            filter_optimizer.zero_grad()
            loss.backward()
            filter_optimizer.step()
            sparse_layer.normalize_weights()
            
        epoch_end = time.perf_counter()    
        epoch_loss /= len(train_loader.sampler)

        if epoch_loss < best_so_far:
            print("found better model")
            # Save model parameters
            torch.save({
                'model_state_dict': sparse_layer.state_dict(),
                'optimizer_state_dict': filter_optimizer.state_dict(),
            }, os.path.join(output_dir, "sparse_conv3d_model-best.pt"))
            best_so_far = epoch_loss

        loss_log.append(epoch_loss)
        print('epoch={}, epoch_loss={:.2f}, time={:.2f}'.format(epoch, epoch_loss, epoch_end - epoch_start))

#     activations = sparse_layer(example_data)
#     reconstructions = sparse_layer.reconstructions(
#         activations).cpu().detach().numpy()

#     print("SHAPES")
#     print(example_data.shape)
#     print(example_data.shape)

#     fig = plt.figure()

#     img_to_show = 3
#     for i in range(img_to_show):
#         # original
#         plt.subplot(img_to_show, 2, i*2 + 1)
#         plt.tight_layout()
#         plt.imshow(example_data[i, 0, :, :], cmap='gray',
#                    interpolation='none')
#         plt.title("Original Image\nGround Truth: {}".format(
#             example_targets[0]))
#         plt.xticks([])
#         plt.yticks([])

#         # reconstruction
#         plt.subplot(img_to_show, 2, i*2 + 2)
#         plt.tight_layout()
#         plt.imshow(reconstructions[i, 0, :, :], cmap='gray',
#                    interpolation='none')
#         plt.title("Reconstruction")
#         plt.xticks([])
#         plt.yticks([])

#     plt.show()

#     plot_filters(sparse_layer.filters.cpu().detach())
