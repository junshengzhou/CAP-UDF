# -*- coding: utf-8 -*-

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.dataset import Dataset
from models.fields import CAPUDFNetwork
import argparse
from pyhocon import ConfigFactory
import os
from shutil import copyfile
import numpy as np
import trimesh
from tools.logger import get_logger, get_root_logger, print_log
from tools.utils import remove_far, remove_outlier
from tools.surface_extraction import as_mesh, surface_extraction
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import math
from scipy.spatial import cKDTree
import point_cloud_utils as pcu
import csv
import warnings
warnings.filterwarnings('ignore')

def extract_fields(bound_min, bound_max, resolution, query_func, grad_func):
    N = 32
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    g = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)
    # with torch.no_grad():
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)

                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()

                grad = grad_func(pts).reshape(len(xs), len(ys), len(zs), 3).detach().cpu().numpy()
                val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
                g[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = grad

    return u, g

def extract_geometry(bound_min, bound_max, resolution, threshold, out_dir, iter_step, dataname, logger, query_func, grad_func):

    print('Extracting mesh with resolution: {}'.format(resolution))
    u, g = extract_fields(bound_min, bound_max, resolution, query_func, grad_func)
    b_max = bound_max.detach().cpu().numpy()
    b_min = bound_min.detach().cpu().numpy()
    mesh = surface_extraction(u, g, out_dir, iter_step, b_max, b_min, resolution)

    return mesh

class Runner:
    def __init__(self, args, conf_path):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.base_exp_dir = self.conf['general.base_exp_dir'] + args.dir
        os.makedirs(self.base_exp_dir, exist_ok=True)
        
        
        self.dataset = Dataset(self.conf['dataset'], args.dataname)
        self.dataname = args.dataname
        self.iter_step = 0

        # Training parameters
        self.step1_maxiter = self.conf.get_int('train.step1_maxiter')
        self.step2_maxiter = self.conf.get_int('train.step2_maxiter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.batch_size_step2 = self.conf.get_int('train.batch_size_step2')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.eval_num_points = self.conf.get_int('train.eval_num_points')
        self.df_filter = self.conf.get_float('train.df_filter')

        self.ChamferDisL1 = ChamferDistanceL1().cuda()
        self.ChamferDisL2 = ChamferDistanceL2().cuda()

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.model_list = []
        self.writer = None

        # Networks
        self.udf_network = CAPUDFNetwork(**self.conf['model.udf_network']).to(self.device)
        if self.conf.get_string('train.load_ckpt') != 'none':
            self.udf_network.load_state_dict(torch.load(self.conf.get_string('train.load_ckpt'), map_location=self.device)["udf_network_fine"])

        self.optimizer = torch.optim.Adam(self.udf_network.parameters(), lr=self.learning_rate)

        # Backup codes and configs for debug
        self.file_backup()

    def train(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(os.path.join(self.base_exp_dir), f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger
        batch_size = self.batch_size
        batch_size_step2 = self.batch_size_step2

        for iter_i in tqdm(range(self.iter_step, self.step2_maxiter)):
            self.update_learning_rate(self.iter_step)

            if self.iter_step < self.step1_maxiter:
                points, samples, point_gt = self.dataset.get_train_data(batch_size)
            else:
                points, samples, point_gt = self.dataset.get_train_data_step2(batch_size_step2)
                
            samples.requires_grad = True
            gradients_sample = self.udf_network.gradient(samples).squeeze() # 5000x3
            udf_sample = self.udf_network.udf(samples)                      # 5000x1
            grad_norm = F.normalize(gradients_sample, dim=1)                # 5000x3
            sample_moved = samples - grad_norm * udf_sample                 # 5000x3

            loss_cd = self.ChamferDisL1(points.unsqueeze(0), sample_moved.unsqueeze(0))
            
            loss = loss_cd
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.iter_step += 1
            if self.iter_step % self.report_freq == 0:
                print_log('iter:{:8>d} cd_l1 = {} lr={}'.format(self.iter_step, loss_cd, self.optimizer.param_groups[0]['lr']), logger=logger)
            
            if self.iter_step == self.step1_maxiter or self.iter_step == self.step2_maxiter:
                self.save_checkpoint()
            
            if self.iter_step == self.step1_maxiter:
                gen_pointclouds = self.gen_extra_pointcloud(self.iter_step, self.conf.get_float('train.low_range'))
                idx = pcu.downsample_point_cloud_poisson_disk(gen_pointclouds, num_samples=int(self.conf.get_float('train.extra_points_rate')*point_gt.shape[0]))
                poisson_pointclouds = gen_pointclouds[idx]
                dense_pointclouds = np.concatenate((point_gt.detach().cpu().numpy(), poisson_pointclouds))
                self.ptree = cKDTree(dense_pointclouds)
                self.dataset.gen_new_data(self.ptree)
            
            if self.iter_step == self.step2_maxiter:
                gen_pointclouds = self.gen_extra_pointcloud(self.iter_step, 1)

            if self.iter_step == self.step1_maxiter or self.iter_step == self.step2_maxiter: 
                self.extract_mesh(resolution=args.mcube_resolution, threshold=0.0, point_gt=point_gt, iter_step=self.iter_step, logger=logger)


    def extract_mesh(self, resolution=64, threshold=0.0, point_gt=None, iter_step=0, logger=None):

        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        out_dir = os.path.join(self.base_exp_dir, 'mesh')
        os.makedirs(out_dir, exist_ok=True)

        mesh = extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold, \
                out_dir=out_dir, iter_step=iter_step, dataname=self.dataname, logger=logger, \
                query_func=lambda pts: self.udf_network.udf(pts), grad_func=lambda pts: self.udf_network.gradient(pts))
        if self.conf.get_float('train.far') > 0:
            mesh = remove_far(point_gt.detach().cpu().numpy(), mesh, self.conf.get_float('train.far'))

        mesh.export(out_dir+'/'+str(iter_step)+'_mesh.obj')



    def gen_extra_pointcloud(self, iter_step, low_range):

        res = []
        num_points = self.eval_num_points
        gen_nums = 0

        os.makedirs(os.path.join(self.base_exp_dir, 'pointcloud'), exist_ok=True)

        while gen_nums < num_points:
            
            points, samples, point_gt = self.dataset.get_train_data(5000)
            offsets = samples - points
            std = torch.std(offsets)

            extra_std = std * low_range
            rands = torch.normal(0.0, extra_std, size=points.shape)   
            samples = points + torch.tensor(rands).cuda().float()

            samples.requires_grad = True
            gradients_sample = self.udf_network.gradient(samples).squeeze() # 5000x3
            udf_sample = self.udf_network.udf(samples)                      # 5000x1
            grad_norm = F.normalize(gradients_sample, dim=1)                # 5000x3
            sample_moved = samples - grad_norm * udf_sample                 # 5000x3

            index = udf_sample < self.df_filter
            index = index.squeeze(1)
            sample_moved = sample_moved[index]
            
            gen_nums += sample_moved.shape[0]

            res.append(sample_moved.detach().cpu().numpy())

        res = np.concatenate(res)
        res = res[:num_points]
        np.savetxt(os.path.join(self.base_exp_dir, 'pointcloud', 'point_cloud%d.xyz'%(iter_step)), res)

        res = remove_outlier(point_gt.detach().cpu().numpy(), res, dis_trunc=self.conf.get_float('train.outlier'))
        return res

    def update_learning_rate(self, iter_step):

        warn_up = self.warm_up_end
        max_iter = self.step2_maxiter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
        lr = lr * init_lr

        for g in self.optimizer.param_groups:
            g['lr'] = lr
            
    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        print(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))
        self.udf_network.load_state_dict(checkpoint['udf_network_fine'])
        
        self.iter_step = checkpoint['iter_step']
            
    def save_checkpoint(self):
        checkpoint = {
            'udf_network_fine': self.udf_network.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
    
        
if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/ndf.conf')
    parser.add_argument('--mcube_resolution', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dir', type=str, default='test')
    parser.add_argument('--dataname', type=str, default='demo')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args, args.conf)

    runner.train()
