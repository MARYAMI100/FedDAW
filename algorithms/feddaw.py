import copy
import torch
from utils.model import *
from utils.utils import *
from algorithms.client import local_train_net
import math
import numpy as np

def getWeightsBasedOnSimilarity(global_w, net_para):
    flattened_arrays = []
    for key in global_w.keys():
          flattened_arrays.append(global_w[key].cpu().view(-1).numpy())
    global_w_flat = np.concatenate(flattened_arrays)

    flattened_arrays = []
    for key in net_para.keys():
        flattened_arrays.append(net_para[key].cpu().view(-1).numpy())
    client_flat = np.concatenate(flattened_arrays)
    
    dot_product = np.dot(global_w_flat, client_flat)    
    norm_a1 = np.linalg.norm(global_w_flat)
    norm_a2 = np.linalg.norm(client_flat)
    
    return np.abs(dot_product / (norm_a1 * norm_a2))

def getWeightsBasedOnCorentropySimilarity(global_w, net_para):
    flattened_arrays = []
    for key in global_w.keys():
          flattened_arrays.append(global_w[key].cpu().view(-1).numpy())
    global_w_flat = np.concatenate(flattened_arrays)

    flattened_arrays = []
    for key in net_para.keys():
        flattened_arrays.append(net_para[key].view(-1).numpy())
    client_flat = np.concatenate(flattened_arrays)
    
    diff = global_w_flat - client_flat
    sigma = 1
    kernel_values = gaussian_kernel(diff, sigma)
    similarity = np.mean(kernel_values)

def gaussian_kernel(x, sigma):
    return np.exp(-(x**2) / (2 * sigma**2))
    
def feddaw_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, moment_v, device, global_dist, logger):
    best_test_acc=0
    record_test_acc_list = []

    for round in range(n_comm_rounds):
        logger.info("in comm round:" + str(round))
        print("In communication round:" + str(round))
        party_list_this_round = party_list_rounds[round]
        if args.sample_fraction<1.0:
            print(f'Clients this round : {party_list_this_round}')

        global_w = global_model.state_dict()
        if args.server_momentum:
            old_w = copy.deepcopy(global_model.state_dict())

        nets_this_round = {k: nets[k] for k in party_list_this_round}
        for net in nets_this_round.values():
            net.load_state_dict(global_w)
        
        # Local update
        local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device, logger=logger)
        
        # Aggregation weight calculation
        total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
        
        if round==0 or args.sample_fraction<1.0:
            print(f'Dataset size weight : {fed_avg_freqs}')

        sim_weights = np.zeros_like(fed_avg_freqs, dtype=np.float32)
        xi_t = np.zeros_like(fed_avg_freqs, dtype=np.float32)
        for net_id, net in enumerate(nets_this_round.values()):
            if round==0:
                xi_t = np.asarray(fed_avg_freqs, dtype=np.float32)
                break
            else:
                net_para = net.state_dict()
                sim_weights[net_id] = getWeightsBasedOnSimilarity(global_w, net_para)
                #sim_weights[net_id] = getWeightsBasedOnCorentropySimilarity(global_w, net_para)
                etha0 = 0.1
                gamma_t = math.exp(-1*etha0 * round)
                xi_t[net_id] = gamma_t * fed_avg_freqs[net_id] + (1 - gamma_t)*sim_weights[net_id]
        lamb = 0.1
        beta0 = 1
        beta_t = beta0 * math.exp(-1*lamb*round)
        omega = np.exp(beta_t * xi_t) / np.sum(np.exp(beta_t * xi_t))
        
        # Model aggregation
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            
            if net_id == 0:
                for key in net_para:
                    global_w[key] = net_para[key] * omega[net_id]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * omega[net_id]

        if args.server_momentum:
            delta_w = copy.deepcopy(global_w)
            for key in delta_w:
                delta_w[key] = old_w[key] - global_w[key]
                moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                global_w[key] = old_w[key] - moment_v[key]
        global_model.load_state_dict(global_w)
        global_model.cuda()

        # Test
        test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
        record_test_acc_list.append(test_acc)
        global_model.to('cuda')

        if(best_test_acc<test_acc):
            best_test_acc=test_acc
            logger.info('New Best best test acc:%f'% test_acc)
        logger.info('>> Global Model Test accuracy: %f' % test_acc)
        logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
        print('>> Global Model Test accuracy: %f, Best: %f' % (test_acc, best_test_acc))
        
        mkdirs(args.modeldir+'feddaw/')
        if args.save_model:   
            torch.save(global_model.state_dict(), args.modeldir+'feddaw/'+'globalmodel'+args.log_file_name+'.pth')
    return record_test_acc_list, best_test_acc
