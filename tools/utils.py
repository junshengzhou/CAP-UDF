# -*- coding: utf-8 -*-

from random import sample
import time
from tkinter import Variable
from shutil import copyfile
import numpy as np
import trimesh

from scipy.spatial import cKDTree


def get_aver(distances, face):
    return (distances[face[0]] + distances[face[1]] + distances[face[2]]) / 3.0

def remove_far(gt_pts, mesh, dis_trunc=0.1, is_use_prj=False):
    # gt_pts: trimesh
    # mesh: trimesh

    gt_kd_tree = cKDTree(gt_pts)
    distances, vertex_ids = gt_kd_tree.query(mesh.vertices, p=2, distance_upper_bound=dis_trunc)
    faces_remaining = []
    faces = mesh.faces

    if is_use_prj:
        normals = gt_pts.vertex_normals
        closest_points = gt_pts.vertices[vertex_ids]
        closest_normals = normals[vertex_ids]
        direction_from_surface = mesh.vertices - closest_points
        distances = direction_from_surface * closest_normals
        distances = np.sum(distances, axis=1)

    for i in range(faces.shape[0]):
        if get_aver(distances, faces[i]) < dis_trunc:
            faces_remaining.append(faces[i])
    mesh_cleaned = mesh.copy()
    mesh_cleaned.faces = faces_remaining
    mesh_cleaned.remove_unreferenced_vertices()

    return mesh_cleaned

def remove_outlier(gt_pts, q_pts, dis_trunc=0.003, is_use_prj=False):
    # gt_pts: trimesh
    # mesh: trimesh

    gt_kd_tree = cKDTree(gt_pts)
    distances, q_ids = gt_kd_tree.query(q_pts, p=2, distance_upper_bound=dis_trunc)

    q_pts = q_pts[distances<dis_trunc]

    return q_pts

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    Suggested by https://github.com/mikedh/trimesh/issues/507
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        print("is_mesh")
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

# def eval_cd(conf, mesh, dataname, eval_num_points):
#     gtshape = trimesh.load(os.path.join(conf.get_string('dataset.data_dir'), 'ground_truth', dataname+'.ply'))

#     gtshape = as_mesh(gtshape)
#     mesh = as_mesh(mesh)
#     total_size = (gtshape.bounds[1] - gtshape.bounds[0]).max()
#     centers = (gtshape.bounds[1] + gtshape.bounds[0]) / 2
#     gtshape.apply_translation(-centers)
#     gtshape.apply_scale(1 / total_size)

#     total_size_my = (mesh.bounds[1] - mesh.bounds[0]).max()
#     centers_my = (mesh.bounds[1] + mesh.bounds[0]) / 2

#     gt = torch.tensor(gtshape.sample(eval_num_points)).unsqueeze(0).cuda().float()
#     pred = torch.tensor(mesh.sample(eval_num_points)).unsqueeze(0).cuda().float()
#     ChamferDisL2 = ChamferDistanceL2().cuda()
#     cd = ChamferDisL2(gt, pred)*1e4 / 2

#     return cd

# def eval_cd_srb(conf, mesh, dataname, eval_num_points):
#     gtshape = trimesh.load(os.path.join(conf.get_string('dataset.data_dir'), 'ground_truth', dataname+'.ply'))

#     total_size = (gtshape.bounds[1] - gtshape.bounds[0]).max()
#     centers = (gtshape.bounds[1] + gtshape.bounds[0]) / 2

#     mesh.apply_scale(total_size)
#     mesh.apply_translation(centers)

#     gt = torch.tensor(gtshape.vertices).unsqueeze(0).cuda().float()
#     pred = torch.tensor(mesh.sample(eval_num_points)).unsqueeze(0).cuda().float()
#     ChamferDisL1 = ChamferDistanceL1().cuda()
#     cd = ChamferDisL1(gt, pred)

#     mesh.apply_translation(-centers)
#     mesh.apply_scale(1 / total_size)
    
#     return cd
