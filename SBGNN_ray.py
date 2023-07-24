
import argparse
import numpy as np
import os
import networkx as nx
import torch
import ray

import smoothing_based_optimizer
import GNN_core
import time

parser = argparse.ArgumentParser(description="Simulate a Affinity training GNN with the appropriate hyperparameters.")
parser.add_argument('-d','--dataset', required=True, help='the protein dataset')
parser.add_argument('--graph_path', required=True, help='path to the graph files')
parser.add_argument('-r','--partition_ratio', required=False, type=str, help="governs the ration of partition sizes in the training, validation, and test sets. a list of the form [train, val, test]", default="0.4:0.3:0.3")
parser.add_argument('--partition_size', required=False, help='sets partition size for the total size of dataset', default='max')
parser.add_argument('-e','--epochs', required=False, help='number of training epochs', default='201')
parser.add_argument('-nc','--num_layers', required=False, help='number of layers', default='3')
parser.add_argument('-p','--patience', required=False, type=int, help='upper limit for the patience counter used in validation', default=60)
parser.add_argument('-b','--batch_size', required=False, type=int, help='batch size for training, testing and validation', default=30)
parser.add_argument('-l','--learning_rate', required=False, type=float, help='initial learning rate', default=0.008)
parser.add_argument('-m','--model_type', required=False, type=str, help='the underlying model of the neural network', default='GCN')
parser.add_argument('-c','--hidden_channel', required=False, type=int, help='width of hidden layers', default=25)
parser.add_argument('--meta_feature_dim', required=False, help='the output size of the first linear layer or the input dimension of the clustering algorithm', default='same')
parser.add_argument('--cluster_params', required=False, help='p_factor, C_sp and C_degree', default="4,0.8,0.1")
parser.add_argument('--initial_mu', required=False, help='the starting point of the gaussian distribution mu for smoothing based optimization', default="identity")
parser.add_argument('--initial_sigma', required=False, type=float, help='the starting point of the gaussian distribution sigma for smoothing based optimization', default=3.)
parser.add_argument('-o','--params_storage_path',required=False,help='path to store the model params',default=None)

args = parser.parse_args()
protein_dataset=args.dataset
pdb_path=args.graph_path
lr=args.learning_rate
n_epochs=args.epochs
arch=args.model_type
ratio = args.partition_ratio.split(":")
ratio = [float(entry) for entry in ratio]
batch_size=args.batch_size
num_layers=args.num_layers
hidden_channels=args.hidden_channel

meta_feature_dim=args.meta_feature_dim
cluster_params=args.cluster_params.split(",")
cluster_params = [float(entry) for entry in cluster_params]

output_path=args.params_storage_path


### load proteins
proteins_PDB=[]
with open(protein_dataset, "r") as file:
    content = file.read()
for line in content.splitlines():
    line=np.array(list(line.split(" ")))
    proteins_PDB.append(line[0])


if __name__ == '__main__':
    graph_dataset=[]

    for protein_index,my_protein in enumerate(proteins_PDB):
        if os.path.exists(str(pdb_path)+'/'+str(my_protein)+".nx"):

            G = nx.read_gpickle(str(pdb_path)+'/'+str(my_protein)+".nx")
            G.prot_idx = torch.tensor(protein_index,dtype=torch.long)
            graph_dataset.append(G)



    graph_dataset=GNN_core.balance_dataset(graph_dataset)
    graph_dataset=GNN_core.alternate_g(graph_dataset)

    if args.partition_size != 'max':
        graph_dataset=graph_dataset[:int(args.partition_size)]
    GNN_core.get_info_dataset(graph_dataset,verbose=True)

    ### convert to undirect
    for index,g in enumerate(graph_dataset):
        new_edge=[]
        old_edge=g['edge_index']
        old_edge=old_edge.numpy().T
        for e in old_edge:
            new_edge.append(e)
            new_edge.append([e[1],e[0]])
        g['edge_index']=torch.from_numpy(np.array(new_edge).T)

    assert(ratio[0]+ratio[1]+ratio[2]==1)
    part1 = int(len(graph_dataset)*ratio[0])
    part2 = part1 + int(len(graph_dataset)*ratio[1])
    part3 = part2 + int(len(graph_dataset)*ratio[2])

    num_node_features=len(graph_dataset[0].x[0])


    result_degree=[0]*len(graph_dataset)
    result_distance=[0]*len(graph_dataset)
    ray.init()
    BATCHES=len(graph_dataset)
    time3=time.time()
    for i in range(BATCHES):
        result_degree[i]=smoothing_based_optimizer.calc_degree_ray.remote(graph_dataset,i)
        result_distance[i]=smoothing_based_optimizer.calc_distance_ray.remote(graph_dataset,i)
    degree_matrices = ray.get(result_degree)
    distance_matrices = ray.get(result_distance)
    time4=time.time()
    ray.shutdown()
    print('matrices time with ray:',time4-time3)

    if str(meta_feature_dim)=='same':
        meta_feature_dim=len(graph_dataset[0]['x'][0])
    else:
        meta_feature_dim=int(meta_feature_dim)


##############################

    ray.init()
    if args.initial_mu=='identity':
        mu=[0.]*(meta_feature_dim*num_node_features+meta_feature_dim) ##set the initial value to (partial) identity matrix
        for i in range(meta_feature_dim):
            mu[meta_feature_dim+i*num_node_features+i]=1.
    else:
        mu=np.loadtxt(args.initial_mu)

    sigma=args.initial_sigma

    epsilon=0.0001
    t=0
    sigma_list=[]
    val_accu_list=[]
    test_accu_list=[]
    train_accu_list=[]

    ########################################################################################
    # Automatically load last chcekpoint once program is interrupted
    if output_path!=None:
        if os.path.exists(output_path):
            checkpoints = os.listdir(output_path)
            if len(checkpoints) != 0 :
                t = max([int(checkpoint.split('t')[1]) for checkpoint in checkpoints]) - 1
                if t != -1:
                    mu = np.loadtxt(os.path.join(output_path, 't' + str(t), 'mu.txt'))
                    sigma = np.loadtxt(os.path.join(output_path, 't' + str(t), 'sigma.txt'))
                else:
                    t = 0
    ########################################################################################

    while epsilon<sigma and t<60:
        print('\n start with t =',t)
        N_sample=2
        time1=time.time()
        BATCHES=N_sample
        results=[]

        ### update mu
        for _ in range(BATCHES):
            results.append(smoothing_based_optimizer.calc_loss_onePoint.remote(mu,sigma,meta_feature_dim,num_node_features,graph_dataset,part1,part2,batch_size,n_epochs,args.patience,degree_matrices,distance_matrices,hidden_channels,num_layers,arch,lr,cluster_params))
        output = ray.get(results)

        sample=[np.float_(row[0]) for row in output]
        objective_func=np.array([row[1] for row in output])
        new_mu=smoothing_based_optimizer.update_mu(sample,objective_func)

        if type(new_mu)!=np.ndarray and new_mu==None:
            new_mu=mu

        results_1=[]
        for _ in range(BATCHES):
            results_1.append(smoothing_based_optimizer.calc_loss_onePoint.remote(new_mu,sigma,meta_feature_dim,num_node_features,graph_dataset,part1,part2,batch_size,n_epochs,args.patience,degree_matrices,distance_matrices,hidden_channels,num_layers,arch,lr,cluster_params))
        output_1 = ray.get(results_1)

        sample_1=[np.float_(row[0]) for row in output_1]
        objective_func_1=np.array([row[1] for row in output_1])

        new_sigma=smoothing_based_optimizer.update_sigma(sample_1,objective_func_1,current_mu=new_mu)
        if new_sigma==None:
            new_sigma=sigma
        time2=time.time()
        print('calculate all',N_sample,'samples in t =',t,'in time',time2-time1)

        val_accu=[row[2] for row in output_1]
        train_accu=[row[3] for row in output_1]
        test_accu=[row[4] for row in output_1]
        best_model=[row[5] for row in output_1]
        if output_path!=None:
            smoothing_based_optimizer.store_model_params(output_path,sample_1,objective_func_1,best_model,val_accu,train_accu,test_accu,t)

            my_path=str(output_path)+'/t'+str(t)
            with open(my_path+'/mu.txt', 'w') as fp:
                for item in new_mu:
                    fp.write("%s\n" % item)
            with open(my_path+'/sigma.txt', 'w') as fpp:
                fpp.write("%s\n" % new_sigma)
        
        average_val_accu=round(np.mean(val_accu),3)
        average_train_accu=round(np.mean(train_accu),3)
        average_test_accu=round(np.mean(test_accu),3)

        val_accu_list.append(average_val_accu)
        train_accu_list.append(average_train_accu)
        test_accu_list.append(average_test_accu)
        print('all test accuracies:',np.round_(test_accu,2))
        print('sigma:',round(sigma,5))
        print('average val accuracy:',average_val_accu)
        print('average train accuracy:',average_train_accu)
        print('average test accuracy:',average_test_accu)

        sigma=new_sigma
        mu=new_mu
        t+=1
    print('final result!')
    print('final mu',mu)
    print('all mean val',val_accu_list)
    print('all mean test',test_accu_list)
