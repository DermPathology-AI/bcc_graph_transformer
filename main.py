#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function


import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from utils.dataset import GraphDataset
from utils.lr_scheduler import LR_Scheduler
from tensorboardX import SummaryWriter
from helper import Trainer, Evaluator, collate
from option import Options

# from utils.saliency_maps import *

from models.GraphTransformer import Classifier
from models.weight_init import weight_init

import pandas as pd
import numpy as np

args = Options().parse()
n_class = args.n_class

print(os.system('python3 -c "import torch; torch.cuda.get_device_name(0)"'))

torch.cuda.synchronize()
torch.backends.cudnn.deterministic = True

data_path  = args.data_path
model_path = args.model_path
if not os.path.isdir(model_path): os.mkdir(model_path)
log_path = args.log_path
if not os.path.isdir(log_path): os.mkdir(log_path)
task_name = args.task_name


###################################
train = args.train
test = args.test
graphcam = args.graphcam
print("train:", train, "test:", test, "graphcam:", graphcam)

##### Load datasets
print("preparing datasets and dataloaders......")
batch_size = args.batch_size

if train:
    ids_train = open(args.train_set).readlines()    
    dataset_train = GraphDataset(os.path.join(data_path, ""), ids_train)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=10, collate_fn=collate, shuffle=True, pin_memory=True, drop_last=True)
    total_train_num = len(dataloader_train) * batch_size

ids_val = open(args.val_set).readlines()
dataset_val = GraphDataset(os.path.join(data_path, ""), ids_val)
dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=10, collate_fn=collate, shuffle=False, pin_memory=True)
total_val_num = len(dataloader_val) * batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device
print('Using', torch.cuda.device_count(), 'GPUs')


print("creating models......")

num_epochs = args.num_epochs
learning_rate = args.lr

model = Classifier(n_class)

model = nn.DataParallel(model, device_ids=[0]) # parallelise
if args.resume:
    print('load model{}'.format(args.resume))
    model.load_state_dict(torch.load(args.resume))

if torch.cuda.is_available():
    model = model.cuda()
#model.apply(weight_init)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 5e-4)       # best:5e-4, 4e-3
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70], gamma=0.1)    # gamma=0.3  # 30,90,130 # 20,90,130 -> 150

##################################

criterion = nn.CrossEntropyLoss()

if not test:
    #with open('scripts/train_set_0.txt') as f:  f = f.readlines()
    #num_classes = len(set([tv.replace('\n','').split('\t')[1] for tv in f]))
    num_classes = n_class
    writer = SummaryWriter(log_dir=log_path + task_name)
    f_log = open(log_path + task_name + '_' + str(num_classes) + ".log", 'w')

trainer = Trainer(n_class)
evaluator = Evaluator(n_class)

best_pred = 0.0
for epoch in range(num_epochs):
    # optimizer.zero_grad()
    model.train()
    train_loss = 0.
    total = 0.

    current_lr = optimizer.param_groups[0]['lr']
    print('\n=>Epoches %i, learning rate = %.7f, previous best = %.4f' % (epoch+1, current_lr, best_pred))

    if train:
        for i_batch, sample_batched in enumerate(dataloader_train):
            #scheduler(optimizer, i_batch, epoch, best_pred)
            scheduler.step(epoch)

            preds, labels, loss, p = trainer.train(sample_batched, model)

            optimizer.zero_grad()
            #loss.backward()
            loss.sum().backward()  # Filmon added this
            optimizer.step()

            train_loss += loss
            total += len(labels)

            trainer.metrics.update(labels, preds)
            #trainer.plot_cm()

            if (i_batch + 1) % args.log_interval_local == 0:
                print("[%d/%d] train loss: %.3f; agg acc: %.3f" % (total, total_train_num, np.mean(np.array(train_loss.tolist())) / total, trainer.get_scores()))
                #trainer.plot_cm()
                
                #accuracy, auc_value, precision, recall, fscore = five_scores(labels, preds)
                #sys.stdout.write('\r Epoch [%d/%d] train loss: %.4f, test loss: %.4f, accuracy: %.4f, aug score: %.4f, precision: %.4f, recall: %.4f, fscore: %.4f ' % 
                #  (epoch+1, args.num_epoch, train_loss, test_loss, accuracy, auc_value, precision, recall, fscore))
                

    if not test: 
        print("[%d/%d] train loss: %.3f; agg acc: %.3f" % (total_train_num, total_train_num, np.mean(np.array(train_loss.tolist())) / total, trainer.get_scores()))
        trainer.plot_cm()


    if epoch % 1 == 0:
        with torch.no_grad():
            model.eval()
            print("evaluating...")

            total = 0.
            batch_idx = 0
            
            preds_list = []
            labels_list = []
            proba_list = []

            for i_batch, sample_batched in enumerate(dataloader_val):
                #pred, label, _ = evaluator.eval_test(sample_batched, model)
        
                preds, labels, loss, p = evaluator.eval_test(sample_batched, model, graphcam)

                
                total += len(labels)

                evaluator.metrics.update(labels, preds)
                
                labels_list.append(labels.tolist())
                preds_list.append(preds.tolist())
                proba_list.append(p.tolist())
                
                
                if (i_batch + 1) % args.log_interval_local == 0:
                    print('[%d/%d] val agg acc: %.3f' % (total, total_val_num, evaluator.get_scores()))
                    #evaluator.plot_cm()
                    #accuracy, auc_value, precision, recall, fscore = five_scores(labels, preds)
                    #sys.stdout.write('\r accuracy: %.4f, aug score: %.4f, precision: %.4f, recall: %.4f, fscore: %.4f ' % (accuracy, auc_value, precision, recall, fscore))
            
            print('[%d/%d] val agg acc: %.3f' % (total_val_num, total_val_num, evaluator.get_scores()))
            evaluator.plot_cm()
            
            val_set_df = pd.DataFrame()
            val_set_df['label'] = sum(labels_list,[])
            val_set_df['preds'] = sum(preds_list, [])
            val_set_df.astype('object')
            val_set_df['proba'] = list(sum(proba_list, []))
 
            
            val_set_df.to_csv(f'val_test_results/{task_name}.csv')
            
            # torch.cuda.empty_cache()

            val_acc = evaluator.get_scores()
            if val_acc > best_pred: 
                best_pred = val_acc
                if not test:
                    print("saving model...")
                    torch.save(model.state_dict(), model_path + task_name + ".pth")

            log = ""
            log = log + 'epoch [{}/{}] ------ acc: train = {:.4f}, val = {:.4f}'.format(
                                        epoch+1, num_epochs, trainer.get_scores(), evaluator.get_scores()) + "\n"
            log += "================================\n"
            print(log)
            
            log_clean = ""
            log_clean = log_clean + '{} {:.4f} {:.4f}'.format(epoch+1, trainer.get_scores(), evaluator.get_scores()) + '\n'
            
            if test: break

            #f_log.write(log)
            f_log.write(log_clean)
            f_log.flush()

            writer.add_scalars('accuracy', {'train acc': trainer.get_scores(), 'val acc': evaluator.get_scores()}, epoch+1)
            
            # saves the last step
            if epoch==num_epochs-1:
                preds_list_train = []
                labels_list_train = []

                for i_batch, sample_batched in enumerate(dataloader_train):
                    preds_, labels_, loss, p = trainer.train(sample_batched, model)
                    labels_list_train.append(labels_.tolist())
                    preds_list_train.append(preds_.tolist())
                    
                #train_set_df = pd.DataFrame()
                #train_set_df['label'] = sum(labels_list_train,[])
                #train_set_df['preds'] = sum(preds_list_train, [])
                #train_set_df.to_csv('train_set_result.csv')
                    
    #writer.export_scalars_to_json("./all_scalars.json")
    trainer.reset_metrics()
    evaluator.reset_metrics()

if not test: f_log.close()