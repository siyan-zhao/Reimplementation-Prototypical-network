import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils import data
from numpy import array
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
import torch.optim as optim

# Import models


from baseline import Baseline

# Hyperparameters

batch_size = 32
torch.manual_seed(2)
learning_rate = 0.1
criterion = nn.CrossEntropyLoss()

# Data loading and processing

inputs_data = np.load('inputs.npy')
labels_data = np.load('labels.npy')

class encoder(nn.Module):
    def __init__(self, x_dim = 1, hid_dim = 64, z_dim = 64):
        super(encoder,self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

# Distance metric

def few_shot_euclidean_similarity(query, prototypes):

    a = prototypes.size()[0]
    b = query.size()[0]
    # tile vectors to dimension (b, a, emb_dim)
    query = query.unsqueeze(1).repeat(1, a, 1)
    prototypes = prototypes.unsqueeze(0).repeat(b, 1, 1)
    print(query.size(),'q')
    print(prototypes.size(),'p')
    return torch.pow(prototypes - query, 2).mean(2)


def cosine_similarity(x1, x2):
    norm1 = x1.norm(p=2)
    norm2 = x2.norm(p=2)
    return 1 - torch.dot(x1, x2) / (norm1 * norm2)


# Main training loop

if __name__ == "__main__":
    epoch = 20
    num_iteration = 10
    n_examples = 350
    n_shot = 5
    n_way = 5
    n_classes = 8
    n_query = 15
    baseline = False
    emb_dim = 64
    im_height = 128
    im_width = 128
    channels = 3
    if baseline:
        model = Baseline()
    else:
        model = encoder()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    print('here')

    for idx in range(epoch):
        for iter_idx in range(num_iteration):
            # splits data into support and query sets
            epi_classes = np.random.permutation(n_classes)[:n_way]
            support = np.zeros([n_way, n_shot, channels, im_height, im_width], dtype=np.float32)
            query = np.zeros([n_way, n_query, channels, im_height, im_width], dtype=np.float32)
            support_label = np.zeros([n_way * n_shot])
            query_label = np.zeros([n_way * n_query])
            for i, epi_cls in enumerate(epi_classes):
                random = np.random.permutation(n_examples)[:n_shot + n_query]
                support[i] = inputs_data[epi_cls,random[:n_shot]]
                query[i] = inputs_data[epi_cls, random[n_shot:]]
                label_support = np.tile(np.array([i]),n_shot)
                label_query = np.tile(np.array([i]),n_query)
                support_label[n_shot * i : n_shot * (i+1)] = label_support
                query_label[n_query * i : n_query * (i+1)] = label_query
            support = torch.Tensor(support)
            query = torch.Tensor(query)
            support_label = torch.Tensor(support_label)
            query_label = torch.Tensor(query_label)
            support_inputs = support.view(n_way * n_shot,channels,im_height,im_width)
            query_inputs = query.view(n_way * n_query, channels, im_height, im_width)
            support_embedding = model(support_inputs)

            # embedding is of size [batch, num_example_per_class, emb_dim]
            # find prototype of each class by calculating the average

            support_embedding = support_embedding.view(n_way, -1, emb_dim)
            prototypes = torch.mean(support_embedding, 1)

            query_embeddings = model(query_inputs)

            distances = few_shot_euclidean_similarity(query_embeddings, prototypes)

            m = nn.Softmax(dim = 1)
            dist = m(distances)
            log_p_y = dist.view(n_way, n_query, -1)
            support_label = support_label.view(n_way, n_shot, -1)
            one_hot_query = torch.nn.functional.one_hot(query_label.to(torch.int64), n_way)
            one_hot_query = one_hot_query.view(n_way, n_query, -1).float()

            print('after one hot', one_hot_query.size())

            ce_loss = torch.sum(torch.mul(one_hot_query, log_p_y), dim=-1)
            ce_loss = - ce_loss.mean(-1)

            ce_loss.sum().backward()
            optimizer.step()
            query_labels_not_one_hot = torch.argmax(one_hot_query,axis = 2)  # get label idx
            query_labels_not_one_hot = query_labels_not_one_hot.view(n_way, n_query, 1)


            _, predicted = torch.max(log_p_y,-1)
            acc = torch.eq(predicted.unsqueeze(-1), query_labels_not_one_hot).float().mean()
            if(iter_idx % 1 == 0):
                print("Iteration:", iter_idx, "training accuracy: ", acc.item(), "training loss:", ce_loss.sum().item())
