from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
import torch
from torch_geometric.nn import GraphConv,TransformerConv,GCNConv



class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,input_dim,num_classes,num_layers):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv_c=torch.nn.ModuleList()
        self.bn_c=torch.nn.ModuleList()
        for _ in range(int(num_layers)):
            self.conv_c.append(GCNConv(hidden_channels, hidden_channels))
            self.bn_c.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edges, batch):
        x = self.conv1(x, edges)
        x = self.bn1(x)
        x = x.relu()
        if len(self.conv_c) > 0:
            for index,conv_c_i in enumerate(self.conv_c):
                x = conv_c_i(x,edges)
                x = self.bn_c[index](x)
                x = x.relu()
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.lin(x)
        return x

class GTN(torch.nn.Module):
    def __init__(self, hidden_channels,input_dim,num_classes,num_layers):
        super(GTN, self).__init__()
        self.conv1 = TransformerConv(input_dim, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv_c=torch.nn.ModuleList()
        self.bn_c=torch.nn.ModuleList()
        for _ in range(int(num_layers)):
            self.conv_c.append(TransformerConv(hidden_channels, hidden_channels))
            self.bn_c.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edges, batch):
        x = self.conv1(x, edges)
        x = self.bn1(x)
        x = x.relu()
        if len(self.conv_c) > 0:
            for index,conv_c_i in enumerate(self.conv_c):
                x = conv_c_i(x,edges)
                x = self.bn_c[index](x)
                x = x.relu()
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.lin(x)
        return x


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels,input_dim,num_classes,num_layers):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv_c=torch.nn.ModuleList()
        self.bn_c=torch.nn.ModuleList()
        for _ in range(int(num_layers)):
            self.conv_c.append(GraphConv(hidden_channels, hidden_channels))
            self.bn_c.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edges, batch):
        x = self.conv1(x, edges)
        x = self.bn1(x)
        x = x.relu()
        if len(self.conv_c) > 0:
            for index,conv_c_i in enumerate(self.conv_c):
                x = conv_c_i(x,edges)
                x = self.bn_c[index](x)
                x = x.relu()
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.lin(x)
        return x

def train(model,train_loader,optimizer,criterion):
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(model,loader):
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.

def calc_loss(model, loader,criterion):
    model.eval()
    for data in loader:  # Iterate in batches over the training dataset.
        #model.conv1.register_forward_hook(get_activation('conv3'))
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)
    return loss

def predict(model,loader):
    model.eval()
    pred=[]
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        pred.append(out.argmax(dim=1).tolist())  # Use the class with highest probability.
    return pred 


def loss(model,loader,criterion):
    model.eval()
    loss=0.
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y).sum()
    return loss/len(loader.dataset)

def alternate_dataset(dataset):
    alternate=[]
    zeros=[]
    ones=[]
    for i,data in enumerate(dataset):
        label = data.y.item()
        if label == 0:
            zeros.append(i)
        elif label ==1:
            ones.append(i)
    if len(zeros)>len(ones):
        zeros=zeros[0:len(ones)]
    elif len(zeros)<len(ones):
        ones=ones[0:len(zeros)]
    for i,j in zip(zeros,ones):
        alternate.append(dataset[i])
        alternate.append(dataset[j])
    return alternate

def alternate_g(dataset):
    ones, zeros = sort_dataset(dataset)
    return coallated_dataset(ones, zeros)

def sort_dataset(dataset, a_label=1):
    labeled_a = []
    labeled_b = []
    for data in dataset:
        label = data.y.item()
        if label == a_label:
            labeled_a.append(data)
        else:
            labeled_b.append(data)
    return (labeled_a, labeled_b)


def coallated_dataset(set1, set2):
    dataset = []
    if len(set1)<len(set2):
        length = len(set1)
    else:
        length = len(set2)
    for i in range(length):
        dataset.append(set1[i])
        dataset.append(set2[i])
    return dataset

def get_info_dataset(dataset, verbose=False):
    """Determines the number of inputs labeled one and zero in a dataset."""
    zeros = 0
    ones = 0
    for data in dataset:
        label = data.y.item()
        if label == 0:
            zeros+=1
        elif label ==1:
            ones+=1
    if verbose:
        print(f'In this dataset, there are {zeros} inputs labeled "0" and {ones} inputs labeled "1".')
    return (ones, zeros)

def balance_dataset(dataset):
    ones, zeros = get_info_dataset(dataset)
    if zeros==ones:
        return dataset
    if zeros>ones:
        major=zeros
        minor=ones
        the_major_one=0
    else:
        major=ones
        minor=zeros
        the_major_one=1
    major_index=0
    balanced = []
    for item in dataset:
        label = item.y.item()
        if label == the_major_one:
            if major_index<minor:    
                balanced.append(item)
                major_index=major_index+1
        else:
            balanced.append(item)
    return balanced
