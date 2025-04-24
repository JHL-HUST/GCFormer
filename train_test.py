from data import get_dataset


import time
import utils
import random
import numpy as np
import torch
import torch.nn.functional as F
from early_stop import EarlyStopping, Stop_args
from model_mix import SGT
from lr import PolynomialDecayLR
import os.path
import torch.utils.data as Data
import argparse


# Training settings
def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--name', type=str, default="basic")
    parser.add_argument('--dataset', type=str, default='acm',
                        help='Choose from {acm}')
    parser.add_argument('--device', type=int, default=0,
                        help='Device cuda id')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed.')

    # model parameters
    parser.add_argument('--sample_num_p', type=int, default=7,
                        help='Number of node to be sampled')
    parser.add_argument('--sample_num_n', type=int, default=7,
                        help='Number of node to be sampled')

    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden layer size')
    parser.add_argument('--ffn_dim', type=int, default=64,
                        help='FFN layer size')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=1,
                        help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.5,
                        help='Dropout in the attention layer')

    parser.add_argument('--alpha', type=float, default=0.1,
                        help='aggregation weight')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='contrastive loss weight')

    parser.add_argument('--pp_k', type=int, default=3,
                        help='propagation steps')

    parser.add_argument('--temp', type=float, default=10,
                        help='temperature')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--sample_size', type=int, default=50000,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--tot_updates', type=int, default=1000,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=400,
                        help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping')




    return parser.parse_args()




args = parse_args()

print(args)


acc_list = []

device = args.device


random.seed(args.seed) 
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# Load and pre-process data
adj, features, labels, idx_train, idx_val, idx_test = get_dataset(args.dataset)   
print(adj)

data_file = './pre_sample/'+args.dataset +'_'+str(args.sample_num_p)+'_'+str(args.sample_num_n)+"_"+str(args.pp_k)+'.pt'

if os.path.isfile(data_file):
    processed_features = torch.load(data_file)

else:


    processed_features = utils.node_seq_feature(features, args.sample_num_p, args.sample_num_n, args.sample_size)  # return (N, hops+1, d)


    if args.pp_k > 0:

        data_file_ppr = './pre_features'+args.dataset +'_'+str(args.pp_k)+'.pt'

        if os.path.isfile(data_file_ppr):
            ppr_features = torch.load(data_file_ppr)

        else:
            ppr_features = utils.node_neighborhood_feature(adj, features, args.pp_k)  # return (N, d)
            # store the data 
            torch.save(ppr_features, data_file_ppr)

        ppr_processed_features = utils.node_seq_feature(ppr_features, args.sample_num_p, args.sample_num_n, args.sample_size)

        processed_features = torch.concat((processed_features, ppr_processed_features), dim=1)

    # store the data 
    torch.save(processed_features, data_file)





# processed_features   # return (N, sample_num_p+1 + args.sample_num_n+1, d)

labels = labels.to(device)



batch_data_train = Data.TensorDataset(processed_features[idx_train], labels[idx_train])
batch_data_val = Data.TensorDataset(processed_features[idx_val], labels[idx_val])
batch_data_test = Data.TensorDataset(processed_features[idx_test], labels[idx_test])

train_data_loader = Data.DataLoader(batch_data_train, batch_size=args.batch_size, shuffle=True)
val_data_loader = Data.DataLoader(batch_data_val, batch_size=args.batch_size, shuffle=True)
test_data_loader = Data.DataLoader(batch_data_test, batch_size=args.batch_size, shuffle=True)


n_class=labels.max().item() + 1


# model configuration
model = SGT(n_layers=args.n_layers,
            input_dim=processed_features.shape[-1],
            hidden_dim=args.hidden_dim,
            n_class=n_class,
            num_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
            dropout_rate=args.dropout,
            attention_dropout_rate=args.attention_dropout,
            args=args).to(device)

print(model)
print('total params:', sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
lr_scheduler = PolynomialDecayLR(
    optimizer,
    warmup_updates=args.warmup_updates,
    tot_updates=args.tot_updates,
    lr=args.peak_lr,
    end_lr=args.end_lr,
    power=1.0,
)





def train_valid_epoch(epoch):
    model.train()
    loss_train_b = 0
    acc_train_b = 0
    for _, item in enumerate(train_data_loader):
        nodes_features = item[0].to(device)
        labels = item[1].to(device)

        optimizer.zero_grad()


        output, ce_loss, con_loss = model(nodes_features, labels)
        

        loss_train = ce_loss + args.beta * con_loss

        loss_train.backward()
        optimizer.step()
        lr_scheduler.step()

        loss_train_b += loss_train.item()

        acc_train_b += utils.accuracy_batch(output, labels).item()


    model.eval()
    loss_val = 0
    acc_val = 0
    for _, item in enumerate(val_data_loader):
        with torch.no_grad():
            nodes_features = item[0].to(device)
            labels = item[1].to(device)

            output, ce_loss, con_loss = model(nodes_features, labels)
            
            loss_val += ce_loss 


            acc_val += utils.accuracy_batch(output, labels).item()


    print('Epoch: {:04d}'.format(epoch + 1),
        'loss_train: {:.4f}'.format(loss_train_b),
        'acc_train: {:.4f}'.format(acc_train_b / len(idx_train)),
        'loss_val: {:.4f}'.format(loss_val),
        'acc_val: {:.4f}'.format(acc_val / len(idx_val)))

    return loss_val.item(), acc_val


def test():
    loss_test = 0
    acc_test = 0
    
    for _, item in enumerate(test_data_loader):
        with torch.no_grad():
            nodes_features = item[0].to(device)
            labels = item[1].to(device)

            model.eval()

            output, ce_loss, con_loss = model(nodes_features, labels)
            
            loss_test += ce_loss 

        acc_test += utils.accuracy_batch(output, labels).item()

    print("Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test / len(idx_test)))


    return acc_test / len(idx_test)


t_total = time.time()
stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
early_stopping = EarlyStopping(model, **stopping_args)
for epoch in range(args.epochs):
    loss_val, acc_val = train_valid_epoch(epoch)
    if early_stopping.check([acc_val, loss_val], epoch):
        break

print("Optimization Finished!")
print("Train cost: {:.4f}s".format(time.time() - t_total))
# Restore best model
print('Loading {}th epoch'.format(early_stopping.best_epoch + 1))
model.load_state_dict(early_stopping.best_state)

test()

