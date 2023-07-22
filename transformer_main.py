import argparse
from dataread import data_train_raw, data_test_raw, ShipTrajData
import torch
import h5py
import numpy as np
from transformer.Models import Transformer, MLP
from projection_dist import projection_dist
import os
import torch.nn as nn
from torch import optim
from transformer.Optim import ScheduledOptim
from torch.nn import functional as F
import random
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
import time
from torch.utils.data import Dataset, DataLoader
import tqdm
log_writer = SummaryWriter()

os.environ['CUDA_LAUNCH_BLOCKING'] ='1'



base_dir = "D:\实验室\项目\二院\徐老师\算法\\100条航迹\\"

# 加载.mat文件
mat_file = h5py.File(base_dir+'after_track_matrix.mat', 'r')
# 获取文件中的数据集名称
dataset_name = list(mat_file.keys())[0]
# 获取数据集
dataset = mat_file[dataset_name]
# 将数据集转换为numpy数组
after_track = dataset[()]
# 关闭文件
mat_file.close()


mat_file = h5py.File(base_dir+'before_track_matrix.mat', 'r')
dataset_name = list(mat_file.keys())[0]
dataset = mat_file[dataset_name]
before_track = dataset[()]
mat_file.close()

mat_file = h5py.File(base_dir+'miss_track_matrix.mat', 'r')
dataset_name = list(mat_file.keys())[0]
dataset = mat_file[dataset_name]
miss_track = dataset[()]
mat_file.close()

before_track = np.swapaxes(before_track,0,2)
after_track = np.swapaxes(after_track,0,2)
miss_track = np.swapaxes(miss_track,0,2)

x_track = np.concatenate((before_track,after_track),axis=1)

after_track_shuffle = after_track.copy()
random.shuffle(after_track_shuffle)
test_track_incorrect = np.concatenate((before_track,after_track_shuffle),axis=1)
y_track = miss_track.copy()

x_mean = np.mean(x_track[:,:,0],axis=1)
y_mean = np.mean(x_track[:,:,1],axis=1)
z_mean = np.mean(x_track[:,:,2],axis=1)

x_max = np.max(x_track[:,:,0],axis=1)
y_max = np.max(x_track[:,:,1],axis=1)
z_max = np.max(x_track[:,:,2],axis=1)

x_track[:,:,0] = x_track[:,:,0] - x_mean[:,np.newaxis]
x_track[:,:,1] = x_track[:,:,1] - y_mean[:,np.newaxis]
x_track[:,:,2] = x_track[:,:,2] - z_mean[:,np.newaxis]

# x_track[:,:,0] = x_track[:,:,0]/x_max[:,np.newaxis]
# x_track[:,:,1] = x_track[:,:,1]/y_max[:,np.newaxis]
# x_track[:,:,2] = x_track[:,:,2]/z_max[:,np.newaxis]





x_mean_test = np.mean(test_track_incorrect[:,:,0],axis=1)
y_mean_test = np.mean(test_track_incorrect[:,:,1],axis=1)
z_mean_test = np.mean(test_track_incorrect[:,:,2],axis=1)

x_max_test = np.max(test_track_incorrect[:,:,0],axis=1)
y_max_test = np.max(test_track_incorrect[:,:,1],axis=1)
z_max_test = np.max(test_track_incorrect[:,:,2],axis=1)

test_track_incorrect[:,:,0] = test_track_incorrect[:,:,0] - x_mean_test[:,np.newaxis]
test_track_incorrect[:,:,1] = test_track_incorrect[:,:,1] - y_mean_test[:,np.newaxis]
test_track_incorrect[:,:,2] = test_track_incorrect[:,:,2] - z_mean_test[:,np.newaxis]

y_track[:,:,0] = y_track[:,:,0] - x_mean[:,np.newaxis]
y_track[:,:,1] = y_track[:,:,1] - y_mean[:,np.newaxis]
y_track[:,:,2] = y_track[:,:,2] - z_mean[:,np.newaxis]


model_dir = "D:\\english\\casic-2\\model\\transformer\\"


# 创建一个包含1到100的列表
lst = list(range(100))
# 随机排序列表
half = len(lst) // 2
random.shuffle(lst)
lst1 = lst[:half]
lst2 = lst[half:]

loss_function = nn.MSELoss() # 损失函数

def DrawTrajectory(tra_pred,tra_true):
    tra_pred[:,:,0]=tra_pred[:,:,0]*0.00063212+110.12347
    tra_true[:,:,0]=tra_true[:,:,0]*0.00063212+110.12347
    tra_pred[:,:,1]=tra_pred[:,:,1]*0.000989464+20.023834
    tra_true[:,:,1]=tra_true[:,:,1]*0.000989464+20.023834
    idx=random.randrange(0,tra_true.shape[0])
    plt.figure(figsize=(9,6),dpi=150)
    pred=tra_pred[idx,:,:].cpu().detach().numpy()
    true=tra_true[idx,:,:].cpu().detach().numpy()
    np.savetxt('pred_true.txt',np.vstack((pred,true)))
    print("A track includes a total of {0} detection points,and their longtitude and latitude differences are".format(pred.shape[0]))
    for i in range(pred.shape[0]):
        print("{0}:({1} degrees,{2} degrees)".format(i+1,abs(pred[i,0]-true[i,0]),abs(pred[i,1]-true[i,1])))
    print('\n')
    plt.plot(pred[:,0],pred[:,1], "r-o")
    plt.plot(true[:,0],true[:,1], "b-*")
    plt.show()

def cal_performance(tra_pred,tra_true):
    return F.mse_loss(tra_pred,tra_true)


def train(model, data, label, optimizer):
    epochs = 3000
    # model = torch.load(model_dir + 'model.pt')
    for epoch in range(epochs):
        # if epoch!=0:
        optimizer.zero_grad()  # 清零优化器梯度，梯度不清零会一直存在

        # score = score.to(device)
        correct_count = 0

        # pred = model(before_track_data.get(p).to(device), after_track_data.get(q).to(device))

        pre = model(data,label)

        loss = loss_function(pre, label)  # 计算一次损失
        # loss = loss_function(pre_1.float(), data_1.y.float())
        # loss = loss_function(pre_1, data_1.y)

        # loss反向传播就行，这里没有acc监视器
        loss.backward()

        print("epoch: ", epoch, "  loss: ", loss.item())

        # print(" ")

        # 用反向传播得到的梯度进行参数更新
        optimizer.step()
    torch.save(model, model_dir + 'model.pt')


def test(data, label):
    model = torch.load(model_dir + 'model.pt')
    pre = model(data, label)
    loss = loss_function(pre, label)
    print("test  ||   loss: ", loss.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch', type=int, default=30000)
    parser.add_argument('-b', '--batch_size', type=int, default=140)
    parser.add_argument('-d_model', type=int, default=40)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-n_head', type=int, default=2)
    parser.add_argument('-n_layers', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-do_train', type=bool, default=False)
    parser.add_argument('-do_retrain', type=bool, default=False)
    parser.add_argument('-do_eval', type=bool, default=False)
    parser.add_argument('-use_mlp', type=bool, default=False)

    opt = parser.parse_args()
    opt.d_word_vec = opt.d_model
    # device="cuda:0"
    device="cpu"

    transformer = Transformer(
        10000,
        10000,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
    ).to(device)

    mlp = MLP(10,10,25,50,use_extra_input=False).to(device)

    model_train = transformer
    if opt.use_mlp:
        model_train = mlp

    x_track = torch.tensor(x_track).float()
    x_track = x_track.reshape(x_track.shape[0], -1)

    y_track = torch.tensor(y_track).float()
    y_track = y_track.reshape(y_track.shape[0], -1)

    if opt.do_train == True:
        parameters = mlp.parameters() if opt.use_mlp else transformer.parameters()
        # optimizer = ScheduledOptim(
        #     optim.Adam(parameters, betas=(0.9, 0.98), eps=1e-09),
        #     opt.lr, opt.d_model, opt.n_warmup_steps, opt.use_mlp)

        lr = 0.001
        optimizer = torch.optim.Adam(model_train.parameters(), lr=lr)


        if opt.do_retrain == True: # only used for transformer
            checkpoint = torch.load("./checkpoint/ckpt.pth")
            transformer.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])



        train(
            model=model_train,
            data=x_track[lst1,:],
            label=y_track[lst1,:],
            optimizer=optimizer
        )


    if opt.do_eval == True:

        test(
            data=x_track[lst1, :],
            label=y_track[lst1, :]
        )

        test(
            data=x_track[lst2,:],
            label=y_track[lst2, :]
        )

    model = torch.load(model_dir + 'model.pt')
    predict = model(x_track,y_track)

    nums = 10
    my_start = 0
    my_end = 100

    x_track = x_track.reshape(x_track.shape[0],30,3)
    predict = predict.reshape(predict.shape[0], 10, 3)
    predict = predict.detach().numpy()


    x_track[:, :, 0] = x_track[:, :, 0] + x_mean[:, np.newaxis]
    x_track[:, :, 1] = x_track[:, :, 1] + y_mean[:, np.newaxis]
    x_track[:, :, 2] = x_track[:, :, 2] + z_mean[:, np.newaxis]

    # predict[:,:,0] = predict[:,:,0]*x_max[:,np.newaxis]
    # predict[:,:,1] = predict[:,:,1]*y_max[:,np.newaxis]
    # predict[:,:,2] = predict[:,:,2]*z_max[:,np.newaxis]

    predict[:, :, 0] = predict[:, :, 0] + x_mean[:, np.newaxis]
    predict[:, :, 1] = predict[:, :, 1] + y_mean[:, np.newaxis]
    predict[:, :, 2] = predict[:, :, 2] + z_mean[:, np.newaxis]

    # test_track_incorrect[:,:,0] = test_tra ck_incorrect[:,:,0] + x_mean_test[:,np.newaxis]
    # test_track_incorrect[:,:,1] = test_track_incorrect[:,:,1] + y_mean_test[:,np.newaxis]
    # test_track_incorrect[:,:,2] = test_track_incorrect[:,:,2] + z_mean_test[:,np.newaxis]
    #
    # predict_x_test = predict_x_test + x_mean_test[:,np.newaxis]
    # predict_y_test = predict_y_test + y_mean_test[:,np.newaxis]
    # predict_z_test = predict_z_test + z_mean_test[:,np.newaxis]

    plt.figure()
    plt.rcParams['font.family'] = 'Times New Roman'
    ax = plt.axes(projection='3d')

    lst = lst1
    # lst = lst2

    for i in lst:
        x = x_track[i, :, 0]
        y = x_track[i, :, 1]
        z = x_track[i, :, 2]
        ax.plot3D(x, y, z, ".", markersize=1, color="blue")

    for i in lst:
        x = predict[i, :, 0]
        y = predict[i, :, 1]
        z = predict[i, :, 2]
        ax.plot3D(x, y, z, ".", markersize=1, color="red")

    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')
    ax.set_zlabel('z / m')
    #
    # plt.figure()
    # plt.rcParams['font.family'] = 'Times New Roman'
    # ax = plt.axes(projection='3d')
    #
    # for i in lst2[:5]:
    #     x = test_track_incorrect[i,:,0]
    #     y = test_track_incorrect[i,:, 1]
    #     z = test_track_incorrect[i,:, 2]
    #     ax.plot3D(x,y,z,".",markersize=1,color="blue")
    #
    # for i in lst2[:5]:
    #     x = predict_x_test[i,:]
    #     y = predict_y_test[i,:]
    #     z = predict_z_test[i,:]
    #     ax.plot3D(x,y,z,".",markersize=1,color="red")
    #
    # ax.set_xlabel('x / m')
    # ax.set_ylabel('y / m')
    # ax.set_zlabel('z / m')

    plt.show()

    my_dist = projection_dist()

    dist_list = []
    # for i in range(100):
    for i in lst:
        dist = my_dist.distance(miss_track[i, :, :], predict[i, :, :])
        dist_list.append(dist)
    print("dist_list: ", dist_list)
    sio.savemat('D:\\实验室\\项目\\二院\\徐老师\\算法\\transformer-补全\\dist_list.mat', {'dist_list': dist_list})
