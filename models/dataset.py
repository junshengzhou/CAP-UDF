import torch
import torch.nn.functional as F
import numpy as np
import os
from scipy.spatial import cKDTree
import trimesh

def search_nearest_point(point_batch, point_gt):
    num_point_batch, num_point_gt = point_batch.shape[0], point_gt.shape[0]
    point_batch = point_batch.unsqueeze(1).repeat(1, num_point_gt, 1)
    point_gt = point_gt.unsqueeze(0).repeat(num_point_batch, 1, 1)

    distances = torch.sqrt(torch.sum((point_batch-point_gt) ** 2, axis=-1) + 1e-12) 
    dis_idx = torch.argmin(distances, axis=1).detach().cpu().numpy()

    return dis_idx

def process_data(data_dir, dataname):
    if os.path.exists(os.path.join(data_dir, 'input', dataname) + '.ply'):
        pointcloud = trimesh.load(os.path.join(data_dir, 'input', dataname) + '.ply').vertices
        pointcloud = np.asarray(pointcloud)
    elif os.path.exists(os.path.join(data_dir, 'input', dataname) + '.xyz'):
        pointcloud = np.loadtxt(os.path.join(data_dir, 'input', dataname) + '.xyz')
    elif os.path.exists(os.path.join(data_dir, 'input', dataname) + '.npy'):
        pointcloud = np.load(os.path.join(data_dir, 'input', dataname) + '.npy')
    else:
        print('Only support .ply, .xyz or .npy data. Please adjust your data format.')
        exit()
    shape_scale = np.max([np.max(pointcloud[:,0])-np.min(pointcloud[:,0]),np.max(pointcloud[:,1])-np.min(pointcloud[:,1]),np.max(pointcloud[:,2])-np.min(pointcloud[:,2])])
    shape_center = [(np.max(pointcloud[:,0])+np.min(pointcloud[:,0]))/2, (np.max(pointcloud[:,1])+np.min(pointcloud[:,1]))/2, (np.max(pointcloud[:,2])+np.min(pointcloud[:,2]))/2]
    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale

    POINT_NUM = pointcloud.shape[0] // 60
    POINT_NUM_GT = pointcloud.shape[0] // 60 * 60
    QUERY_EACH = 1000000//POINT_NUM_GT

    point_idx = np.random.choice(pointcloud.shape[0], POINT_NUM_GT, replace = False)
    pointcloud = pointcloud[point_idx,:]
    ptree = cKDTree(pointcloud)
    sigmas = []
    for p in np.array_split(pointcloud,100,axis=0):
        d = ptree.query(p,51)
        sigmas.append(d[0][:,-1])
    
    sigmas = np.concatenate(sigmas)
    sample = []
    sample_near = []

    for i in range(QUERY_EACH):
        scale = 0.25 if 0.25 * np.sqrt(POINT_NUM_GT / 20000) < 0.25 else 0.25 * np.sqrt(POINT_NUM_GT / 20000)
        tt = pointcloud + scale*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
        sample.append(tt)
        tt = tt.reshape(-1,POINT_NUM,3)

        sample_near_tmp = []
        for j in range(tt.shape[0]):
            nearest_idx = search_nearest_point(torch.tensor(tt[j]).float().cuda(), torch.tensor(pointcloud).float().cuda())
            nearest_points = pointcloud[nearest_idx]
            nearest_points = np.asarray(nearest_points).reshape(-1,3)
            sample_near_tmp.append(nearest_points)
        sample_near_tmp = np.asarray(sample_near_tmp)
        sample_near_tmp = sample_near_tmp.reshape(-1,3)
        sample_near.append(sample_near_tmp)
        
    sample = np.asarray(sample)
    sample_near = np.asarray(sample_near)

    os.makedirs(os.path.join(data_dir, 'query_data'), exist_ok=True)
    np.savez(os.path.join(data_dir, 'query_data', dataname)+'.npz', sample = sample, point = pointcloud, sample_near = sample_near)

class Dataset:
    def __init__(self, conf, dataname):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.data_name = dataname + '.npz'

        if os.path.exists(os.path.join(self.data_dir, 'query_data', self.data_name)):
            print('Query data existing. Loading data...')
        else:
            print('Query data not found. Processing data...')
            process_data(self.data_dir, dataname)

        load_data = np.load(os.path.join(self.data_dir, 'query_data', self.data_name))
        
        self.point = np.asarray(load_data['sample_near']).reshape(-1,3)
        self.sample = np.asarray(load_data['sample']).reshape(-1,3)
        self.point_gt = np.asarray(load_data['point']).reshape(-1,3)
        self.sample_points_num = self.sample.shape[0]-1

        self.object_bbox_min = np.array([np.min(self.point[:,0]), np.min(self.point[:,1]), np.min(self.point[:,2])]) -0.05
        self.object_bbox_max = np.array([np.max(self.point[:,0]), np.max(self.point[:,1]), np.max(self.point[:,2])]) +0.05
        print('bd:',self.object_bbox_min,self.object_bbox_max)

        self.point = torch.from_numpy(self.point).to(self.device).float()
        self.sample = torch.from_numpy(self.sample).to(self.device).float()
        self.point_gt = torch.from_numpy(self.point_gt).to(self.device).float()
        
        print('NP Load data: End')

    def get_train_data(self, batch_size):
        index_coarse = np.random.choice(10, 1)
        index_fine = np.random.choice(self.sample_points_num//10, batch_size, replace = False)
        index = index_fine * 10 + index_coarse  # for accelerating random choice operation
        points = self.point[index]
        sample = self.sample[index]
        return points, sample, self.point_gt

    def gen_new_data(self, tree):
        distance, index = tree.query(self.sample.detach().cpu().numpy(), 1)
        self.point_new = tree.data[index]
        self.point_new = torch.from_numpy(self.point_new).to(self.device).float()


    def get_train_data_step2(self, batch_size):
        index_coarse = np.random.choice(10, 1)
        index_fine = np.random.choice(self.sample_points_num//10, batch_size, replace = False)
        index = index_fine * 10 + index_coarse  # for accelerating random choice operation
        points = self.point_new[index]
        sample = self.sample[index]
        return points, sample, self.point_gt
    
