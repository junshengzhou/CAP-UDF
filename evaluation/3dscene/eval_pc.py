import argparse
import sys
import numpy as np
import torch
from mesh_evaluator import MeshEvaluator
import trimesh
import os
sys.path.append('tools')
from utils import remove_far
from pyhocon import ConfigFactory

gt_cache={}

def evaluate_pcs(gt_path, es_path, gt_points, n_points=1000000, normalize=False):
    es_pcs = np.loadtxt(es_path)

    gt_scale = 1.
    gt_mesh = trimesh.load_mesh(gt_path)

    # sample from gt mesh
    if gt_path in gt_cache.keys():
        gt_pointcloud, gt_normals, gt_scale = gt_cache[gt_path]
    else:
        if os.path.splitext(gt_path)[-1]=='.xyz': # DGP dataset
            gt_pointcloud = np.loadtxt(gt_path).astype(np.float32)
            gt_normals = None
        else:
            gt_mesh = trimesh.load_mesh(gt_path)
            if isinstance(gt_mesh, trimesh.Trimesh):
                gt_pointcloud, idx = gt_mesh.sample(n_points, return_index=True)
                gt_pointcloud = gt_pointcloud.astype(np.float32)
                gt_normals = gt_mesh.face_normals[idx]
            elif isinstance(gt_mesh, trimesh.PointCloud):
                gt_pointcloud = gt_mesh.vertices.astype(np.float32)
                np.random.shuffle(gt_pointcloud)
                gt_pointcloud = gt_pointcloud[:n_points]
                gt_normals = None
            else:
                raise RuntimeError('Unknown data type!')
        # normalize according to the scale of the gt point cloud
        if normalize:
            center = gt_pointcloud.mean(0)
            gt_scale = np.abs(gt_pointcloud-center).max()
            print('Normalize with scale: %.2f' % gt_scale)
        gt_pointcloud /= gt_scale
        gt_cache.update({gt_path: (gt_pointcloud, gt_normals, gt_scale)})

    total_size = (gt_mesh.bounds[1] - gt_mesh.bounds[0]).max()
    centers = (gt_mesh.bounds[1] + gt_mesh.bounds[0]) / 2
    es_pcs *= total_size
    es_pcs += centers

    result=evaluator.eval_pointcloud(es_pcs, gt_pointcloud, gt_normals)

    print("chamferl2:%.3f"%(result['chamfer-L2']*1000))
    print("chamferl1:%.3f"%result['chamfer-L1'])
    return result


n_points=1000000
evaluator=MeshEvaluator(n_points)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/ndf.conf')
    parser.add_argument('--mcube_resolution', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dir', type=str, default='test')
    parser.add_argument('--dataname', type=str, default='demo')
    parser.add_argument('--pred_mesh', type=str, default='demo')

    args = parser.parse_args()
    conf_path = args.conf
    f = open(conf_path)
    conf_text = f.read()
    f.close()

    conf = ConfigFactory.parse_string(conf_text)
    evaluator=MeshEvaluator(n_points=1000000)

    dataname_split = args.dataname.split('_', 1)[0]

    gt_file = os.path.join(conf.get_string('dataset.data_dir'), 'ground_truth', dataname_split+'.ply')
    pred_file = os.path.join(conf.get_string('general.base_exp_dir'), args.dir, 'pointcloud', 'point_cloud'+str(conf.get_int('train.step2_maxiter'))+'.xyz')
    input_file = os.path.join(conf.get_string('dataset.data_dir'), 'input', args.dataname+'.xyz.npy')

    evaluate_pcs(gt_file, pred_file, np.load(input_file), 1000000)