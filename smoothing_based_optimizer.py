import numpy as np
import AffinityClustering
import GNN_core
import copy
import ray
import torch
from torch_geometric.loader import DataLoader
import os
import json

def gaussian_sampling(mu,sigma):
    cov=np.zeros((len(mu),len(mu)))
    np.fill_diagonal(cov,sigma)
    samples=np.random.multivariate_normal(mu,cov)
    return samples

def update_mu(sample,objective_func):
    norm=0
    mu=0
    for x,f_x in zip(sample,objective_func):
        mu=mu+x*f_x
        norm=norm+f_x
    if norm==0:
        mu=None
    else:
        mu=mu/norm
    return mu

def update_sigma(sample,objective_func,current_mu):
    norm=0
    sigma=0
    for x,f_x in zip(sample,objective_func):
        sigma=sigma+sum((x-current_mu)*(x-current_mu))*f_x/len(x)
        norm=norm+f_x
    if norm==0:
        sigma=None
    else:
        sigma=sigma/norm
    return sigma

def update_mu_sigma(sample,objective_func,current_mu):
    mu=0
    sigma=0
    norm=0
    for x,f_x in zip(sample,objective_func):
        mu=mu+x*f_x
        # sum_i (x_i-mu_i)^2
        sigma=sigma+sum((x-current_mu)*(x-current_mu))*f_x/len(x)
        norm=norm+f_x
    mu=mu/norm  # mu is a vector
    sigma=sigma/norm #sigma is a scalar

    return mu,sigma

def train_GNN(model, train_loader,val_loader,test_loader,optimizer,criterion,n_epochs,patience):

    best_val_acc = -0.1
    best_val_epoch = 0
    best_model=None

    for epoch in range(0, int(n_epochs)):
        GNN_core.train(model=model,train_loader=train_loader,optimizer=optimizer,criterion=criterion)
        #train_acc = GNN_core.test(model=model,loader=train_loader)
        this_val_acc = GNN_core.test(model=model,loader=val_loader)

        #if epoch%5==0:
        #    print('epoch:',epoch,'train acc: ',train_acc,'val acc: ',this_val_acc,'best val acc:',best_val_acc)

        if this_val_acc > best_val_acc: #validation wrapper
            best_val_epoch = epoch
            best_val_acc=this_val_acc
            best_model= copy.deepcopy(model)
            patience_counter = 0
        else:
            patience_counter+=1
        if patience_counter == patience:
            #print("ran out of patience")
            break
    train_acc_best_model=GNN_core.test(model=best_model,loader=train_loader)
    test_acc_best_model=GNN_core.test(model=best_model,loader=test_loader)
    #print('best epoch:',best_val_epoch,'train acc: ',train_acc_best_model,'test acc',test_acc_best_model,'best val acc:',best_val_acc)

    train_loss=GNN_core.calc_loss(model=best_model,loader=train_loader,criterion=criterion)
    print('train loss',train_loss)
    score= -train_loss.item()+0.6
    if score<0:
        score=0
    return score,best_val_acc,train_acc_best_model,test_acc_best_model,best_model

@ray.remote
def calc_loss_onePoint(mu,sigma,meta_feature_dim,num_node_features,graph_dataset,part1,part2,batch_size,n_epochs,patience,degree_matrices,distance_matrices,hidden_channels,num_layers,arch,lr,cluster_params):
    num_classes=2
    if arch == 'GCN':
        model = GNN_core.GCN(hidden_channels,input_dim=meta_feature_dim,num_classes=num_classes,num_layers=num_layers)
    if arch == 'GTN':
        model = GNN_core.GTN(hidden_channels,input_dim=meta_feature_dim,num_classes=num_classes,num_layers=num_layers)
    if arch == 'GNN':
        model = GNN_core.GNN(hidden_channels,input_dim=meta_feature_dim,num_classes=num_classes,num_layers=num_layers)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    preLin_param=gaussian_sampling(mu,sigma)

    pre_lin_bias=preLin_param[:meta_feature_dim]
    pre_lin_weight=np.reshape(preLin_param[meta_feature_dim:], (meta_feature_dim,num_node_features))

    for index,g in enumerate(graph_dataset):
        X_m=np.zeros((len(g['x']),len(g['x'][0])))
        X=g['x'].numpy()
        X_w=np.einsum("ij,kj->ik", X, pre_lin_weight)  
        X_m=X_w+pre_lin_bias
        new_edges=AffinityClustering.AffinityClustering_oneGraph(X_m,g['edge_index'],degree_matrices[index],distance_matrices[index],cluster_params)
        g['edge_index']=new_edges  

    degree_list=[]
    for g in graph_dataset:
        node_degree=len(g['edge_index'][0])/len(g['x'])
        degree_list.append(node_degree)
    print(np.mean(np.array(degree_list)),len(degree_list))

    train_dataset = graph_dataset[:part1]
    test_dataset = graph_dataset[part1:part2]
    val_dataset = graph_dataset[part2:]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss,best_val_acc,train_acc_best_model,test_acc_best_model,best_model=train_GNN(model=model, train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,optimizer=optimizer,criterion=criterion,n_epochs=n_epochs,patience=patience)
    return [list(preLin_param),test_loss,best_val_acc,train_acc_best_model,test_acc_best_model,best_model]

@ray.remote
def calc_degree_ray(graph_dataset,i):
    graph=graph_dataset[i]
    edge_first=graph['edge_index'].detach().numpy()[0]
    edge_second=graph['edge_index'].detach().numpy()[1]
    graph_dict=AffinityClustering.graph_dictionary(edge_first,edge_second,len(graph['x']))
    degree_matrix=AffinityClustering.calc_degree_matrix(graph_dict)
    return degree_matrix

@ray.remote
def calc_distance_ray(graph_dataset,i):
    graph=graph_dataset[i]
    edge_first=graph['edge_index'].detach().numpy()[0]
    edge_second=graph['edge_index'].detach().numpy()[1]
    graph_dict=AffinityClustering.graph_dictionary(edge_first,edge_second,len(graph['x']))
    distance_matrix=AffinityClustering.calc_distance_matrix(graph_dict)
    return distance_matrix


def store_model_params(path,sample,objective_func,models,val_accu,train_accu,test_accu,t):
    N_sample=len(objective_func)
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    my_path=str(path)+'/t'+str(t)
    isExist = os.path.exists(my_path)
    if not isExist:
        os.makedirs(my_path)
    for index in range(N_sample):
        my_dict={}
        my_dict["preLin_params"]=list(sample[index])
        my_dict["objective_func"]=objective_func[index]
        my_dict["train_accu"]=train_accu[index]
        my_dict["val_accu"]=val_accu[index]
        my_dict["test_accu"]=test_accu[index]
        print('my_dict',my_dict)


        with open(my_path+'/sample'+str(index)+'_params.txt', "w") as fp:
            json.dump(my_dict, fp)
        torch.save(models[index], my_path+'/sample'+str(index)+'_GNNmodel.pt')



def load_gnn_model(train_loader,model_path):
    loaded_model = torch.load(model_path)
    score=GNN_core.test(model=loaded_model,loader=train_loader)
    