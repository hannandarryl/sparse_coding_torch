import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from conv_model import ConvLayer
from tqdm import tqdm
import argparse
import torchvision
import os
from load_data import load_bamc_data, load_covid_data

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

def plot_video(video):

    fig = plt.gcf()
    ax = plt.gca()
    
    DPI = fig.get_dpi()
    fig.set_size_inches(video.shape[2]/float(DPI), video.shape[3]/float(DPI))

    ax.set_title("Video")

    T = video.shape[1]
    im = ax.imshow(video[0, 0, :, :],
                     cmap=cm.Greys_r)

    def update(i):
        t = i % T
        im.set_data(video[0, t, :, :])

    return FuncAnimation(plt.gcf(), update, interval=1000/20)

def plot_original_vs_recon(original, reconstruction, idx=0):

    # create two subplots
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.set_title("Original")
    ax2.set_title("Reconstruction")

    T = original.shape[2]
    im1 = ax1.imshow(original[idx, 0, 0, :, :],
                     cmap=cm.Greys_r)
    im2 = ax2.imshow(reconstruction[idx, 0, 0, :, :],
                     cmap=cm.Greys_r)

    def update(i):
        t = i % T
        im1.set_data(original[idx, 0, t, :, :])
        im2.set_data(reconstruction[idx, 0, t, :, :])

    return FuncAnimation(plt.gcf(), update, interval=1000/30)


def plot_filters_3d(filters):
    num_filters = filters.shape[0]
    ncol = 3
    # ncol = int(np.sqrt(num_filters))
    # nrow = int(np.sqrt(num_filters))
    T = filters.shape[2]
    
    if num_filters // ncol == num_filters / ncol:
        nrow = num_filters // ncol
    else:
        nrow = num_filters // ncol + 1

    fig, axes = plt.subplots(ncols=ncol, nrows=nrow,
                             constrained_layout=True,
                             figsize=(ncol*2, nrow*2))

    ims = {}
    for i in range(num_filters):
        r = i // ncol
        c = i % ncol
        ims[(r, c)] = axes[r, c].imshow(filters[i, 0, 0, :, :],
                                        cmap=cm.Greys_r)

    def update(i):
        t = i % T
        for i in range(num_filters):
            r = i // ncol
            c = i % ncol
            ims[(r, c)].set_data(filters[i, 0, t, :, :])

    return FuncAnimation(plt.gcf(), update, interval=1000/20)

def plot_filters_2d(filters):
    num_filters = filters.shape[0]
    ncol = int(np.ceil(np.sqrt(num_filters)))
    nrow = int(np.ceil(np.sqrt(num_filters)))

    fig, axes = plt.subplots(ncols=ncol, nrows=nrow,
                             constrained_layout=True)

    ims = {}
    for i in range(num_filters):
        r = i // ncol
        c = i % ncol
        ims[(r, c)] = axes[r, c].imshow(filters[i, 0, :, :], cmap=cm.Greys_r)

    return plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--kernel_height', default=16, type=int)
    parser.add_argument('--kernel_width', default=16, type=int)
    parser.add_argument('--kernel_depth', default=8, type=int)
    parser.add_argument('--num_kernels', default=32, type=int)
    parser.add_argument('--stride', default=2, type=int)
    parser.add_argument('--conv_dim', default=3, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--dataset', default='bamc', type=str, choices=['bamc', 'covid', 'mnist'])
    
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

    if args.dataset == 'bamc':
        train_loader, test_loader = load_bamc_data(batch_size, 1.0)
        print('Loaded', len(train_loader), 'train examples')
    elif args.dataset == 'covid':
        train_loader, test_loader = load_covid_data(batch_size, train_ratio=1.0)
        print('Loaded', len(train_loader), 'train examples')
    else:
        train_loader = load_mnist_data(batch_size)                               
        print('Loaded', len(train_loader), 'train examples')

    example_data = next(iter(train_loader))

    layer = ConvLayer(in_channels=1,
                               out_channels=args.num_kernels,
                               kernel_size=(args.kernel_depth, args.kernel_height, args.kernel_width),
                               stride=args.stride,
                               padding=0,
                               convo_dim=args.conv_dim)
    model = layer
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(device)

    learning_rate = args.lr
    filter_optimizer = torch.optim.Adam(layer.parameters(),
                                        lr=learning_rate)

    loss_log = []
    best_so_far = float('inf')

    for epoch in tqdm(range(args.epochs)):
        epoch_loss = 0
        epoch_start = time.perf_counter()

        for local_labels, local_batch in train_loader:
            local_batch = local_batch.to(device)
            
            recon = model(local_batch)
            loss = layer.loss(local_batch, recon)
            
            epoch_loss += loss.item() * local_batch.size(0)

            filter_optimizer.zero_grad()
            loss.backward()
            filter_optimizer.step()

        epoch_end = time.perf_counter()    
        epoch_loss /= len(train_loader.sampler)

        if epoch_loss < best_so_far:
            print("found better model")
            # Save model parameters
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': filter_optimizer.state_dict(),
            }, os.path.join(output_dir, "sparse_conv3d_model-best.pt"))
            best_so_far = epoch_loss

        loss_log.append(epoch_loss)
        print('epoch={}, epoch_loss={:.2f}, time={:.2f}'.format(epoch, epoch_loss, epoch_end - epoch_start))
        
    plt.plot(loss_log)
    
    plt.savefig(os.path.join(output_dir, 'loss_graph.png'))

#     activations = sparse_layer(example_data[:1])
#     reconstructions = sparse_layer.reconstructions(
#         activations).cpu().detach().numpy()

#     plot_original_vs_recon(example_data, reconstructions, idx=0)
#     plot_filters(sparse_layer.filters.cpu().detach())
