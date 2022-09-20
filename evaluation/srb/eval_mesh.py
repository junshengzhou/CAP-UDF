import argparse
import numpy as np
import torch
from mesh_evaluator import MeshEvaluator
import trimesh
from pyhocon import ConfigFactory
import os

gt_cache={}

def evaluate_mesh(gt_path, es_path, input_path, n_points=100000, normalize=True):
    es_mesh = trimesh.load_mesh(es_path)
    gt_scale = 1.

    inputshape = trimesh.load(input_path)
    total_size = (inputshape.bounds[1] - inputshape.bounds[0]).max()
    centers = (inputshape.bounds[1] + inputshape.bounds[0]) / 2
    es_mesh.apply_scale(total_size)
    es_mesh.apply_translation(centers)

    # sample from gt mesh
    if gt_path in gt_cache.keys():
        gt_pointcloud, gt_normals, gt_scale = gt_cache[gt_path]
    else:
        if os.path.splitext(gt_path)[-1]=='.xyz': 
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

    scale = np.abs(gt_pointcloud).max()
    if normalize:
        es_mesh.vertices /= gt_scale
        scale = 1.
    thresholds=np.linspace(1./1000, 1, 1000) * scale # for F-Score calculation
    result=evaluator.eval_mesh(es_mesh, gt_pointcloud, gt_normals, thresholds=thresholds)
    print('Chamfer-L1:', result['chamfer-L1'] * 10)
    print('F-score:', result['f-score'])
    return result


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
    evaluator=MeshEvaluator(n_points=50000)

    gt_file = os.path.join(conf.get_string('dataset.data_dir'), 'ground_truth', args.dataname+'.ply')
    pred_file = os.path.join(conf.get_string('general.base_exp_dir'), args.dir, 'mesh', str(conf.get_int('train.step2_maxiter'))+'_mesh.obj')
    input_file = os.path.join(conf.get_string('dataset.data_dir'), 'input', args.dataname+'.ply')

    evaluate_mesh(gt_file, pred_file, input_file, 150000)
