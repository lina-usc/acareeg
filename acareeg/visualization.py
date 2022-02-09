# Authors: Christian O'Reilly <christian.oreilly@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
from mne.externals.pymatreader import read_mat
from mne.transforms import apply_trans
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.spatial import KDTree
from nibabel.freesurfer.io import read_geometry
import trimesh
import matplotlib as mpl
import matplotlib.cm as cm
import open3d
import mne

try:
    import pyrender
except:
    pass
from pathlib import Path
import os

from .infantmodels import get_bem_artifacts


def get_plotting_meshes(amplitudes, vertices, age=None, template=None, norm=None, cmap=None):
    if template is None:
        if age is not None:
            template = f"ANTS{age}-0Months3T"
        else:
            raise ValueError("The age or the template must be specified.")

    montage, trans, bem_model, bem_solution, surface_src = get_bem_artifacts(template)
    df = split_vertices_by_hemi(vertices, amplitudes, surface_src)

    vertices, triangles = get_pial_meshes(6)

    hemi_dict = {"lh": 0, "rh": 1}
    if norm is None:
        norm = mpl.colors.Normalize(vmin=min(df["lh"].amplitude.min(), df["rh"].amplitude.min()),
                                    vmax=max(df["lh"].amplitude.max(), df["rh"].amplitude.max()))
    if cmap is None:
        cmap = plt.get_cmap("Reds")

    meshes = {}
    for hemi in hemi_dict:
        vertno = surface_src[hemi_dict[hemi]]["vertno"]
        rr = surface_src[hemi_dict[hemi]]["rr"]
        pos = rr[vertno[df[hemi].vertice.values]] * 1000
        meshes[hemi] = trimesh.Trimesh(vertices=vertices[hemi], faces=triangles[hemi])
        points = KDTree(pos).query(meshes[hemi].vertices, 1)[1]

        colors = plt.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(df[hemi].amplitude.values[points])
        meshes[hemi].visual.vertex_colors = np.round(colors * 255).astype(np.uint8)

    return meshes


def split_vertices_by_hemi(vertices, amplitudes, surface_src):
    offset = surface_src[0]["nuse"]

    return {"lh": pd.DataFrame(dict(vertice=vertices[vertices < offset],
                                    amplitude=amplitudes[vertices < offset])),
            "rh": pd.DataFrame(dict(vertice=vertices[vertices >= offset] - offset,
                                    amplitude=amplitudes[vertices >= offset]))}


def get_atlas_info(age=None, template=None, subjects_dir=None):

    if template is None:
        if age is not None:
            template = f"ANTS{age}-0Months3T"
        else:
            raise ValueError("The age or the template must be specified.")

    if subjects_dir is None:
        subjects_dir = Path(os.environ["SUBJECTS_DIR"])
    atlas = subjects_dir / template / "mri" / "aparc+aseg.mgz"

    epi_img = nib.load(str(atlas))
    epi_img_data = epi_img.get_fdata()

    # zeroing non cortical regions
    name_regions = mne.source_space._get_lut()["name"]
    no_regions = mne.source_space._get_lut()["id"]
    cortical_nos = [no for no, name in zip(no_regions, name_regions) if "ctx" in name]
    for no in np.unique(epi_img_data):
        if no not in cortical_nos:
            epi_img_data[epi_img_data == no] = 0

    vox2ras_tkr = epi_img.header.get_vox2ras_tkr()
    vox2ras = epi_img.header.get_vox2ras()

    pos_atlas = apply_trans(vox2ras_tkr, np.array(np.where(epi_img_data)).T)

    kdtree = KDTree(pos_atlas)

    return kdtree, vox2ras_tkr, vox2ras, epi_img_data


def get_pial_meshes(age=None, template=None, face_count=20000, subjects_dir=None):

    if template is None:
        if age is not None:
            template = f"ANTS{age}-0Months3T"
        else:
            raise ValueError("The age or the template must be specified.")

    if subjects_dir is None:
        subjects_dir = Path(os.environ["SUBJECTS_DIR"])

    mesh_pattern = "{}/{}/surf/{}.pial"
    vertices = {}
    faces = {}
    for hemi in ["lh", "rh"]:
        vertices_hemi, faces_hemi = read_geometry(mesh_pattern.format(subjects_dir, template, hemi))

        open3d_mesh = open3d.geometry.TriangleMesh(vertices=open3d.utility.Vector3dVector(vertices_hemi),
                                                   triangles=open3d.utility.Vector3iVector(faces_hemi))

        mesh = open3d_mesh.simplify_quadric_decimation(int(face_count/2))
        vertices[hemi] = np.asarray(mesh.vertices)
        faces[hemi] = np.asarray(mesh.triangles)

    return vertices, faces


def get_source_model(template, source_model, subjects_dir=None):
    path = "/home/christian/synchedin/infants_atlas_modeling/fieldtrip/single_subject_analysis/"
    source_model_dict = read_mat(path + "sourcemodel_{}_{}.mat".format(source_model, template))["sourcemodel"]

    kdtree, vox2ras_tkr, vox2ras, epi_img_data = get_atlas_info(template, subjects_dir=subjects_dir)
    source_model_dict["pos"] += (vox2ras_tkr - vox2ras)[:3, 3]

    return source_model_dict


def get_template_source_meshes(template_dipole_df, template):
    vertices, triangles = get_pial_meshes(template, face_count=20000)

    meshes = {}
    for model_type, df in template_dipole_df.groupby(["head_model", "source_model"]):
        head_model, source_model = model_type

        source_model_dict = get_source_model(template, source_model)
        inside = source_model_dict["inside"]
        sourcemodel_kdtree = KDTree(source_model_dict["pos"][inside, :])

        meshes[model_type] = trimesh.Trimesh(vertices=vertices, faces=triangles)
        points = sourcemodel_kdtree.query(meshes[model_type].vertices, 1)[1]
        # Trimesh does not preserve vertice order. So the following line cannot be used:
        # points = sourcemodel_kdtree.query(vertices, 1)[1]
        # assert(np.all(meshes[model_type].vertices == vertices))

        mean_ersp = df.groupby("vertice").mean()["ersp"].sort_index().values[inside]

        # xmin = np.percentile(mean_ersp, 1)
        # xmax = np.percentile(mean_ersp, 99)
        # norm = mpl.colors.Normalize(vmin=xmin, vmax=xmax)
        perc = np.percentile(mean_ersp, np.linspace(75, 100, 100))
        mean_ersp = np.digitize(mean_ersp, perc)
        norm = mpl.colors.Normalize(vmin=mean_ersp.min(), vmax=mean_ersp.max())

        cmap = plt.get_cmap("Reds")
        colors = cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(mean_ersp[points])
        meshes[model_type].visual.vertex_colors = np.round(colors * 255).astype(np.uint8)

    return meshes


def show_meshes(meshes, angle_x=-0.7854, angle_y=0, angle_z=0.31416, 
                ax=None, resolution=(1200, 1200), interactive=False):

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    scene = pyrender.Scene(ambient_light=[0.0, 0.0, 0.0],
                           bg_color=[1.0, 1.0, 1.0], )

    for mesh in meshes:
        mesh = mesh.copy()
        re = trimesh.transformations.euler_matrix(angle_x, angle_y, angle_z, 'rxyz')
        mesh.apply_transform(re)
        scene.add(pyrender.Mesh.from_trimesh(mesh))
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0, 0, 240]
    scene.add(camera, pose=camera_pose)

    ligth_poses = [np.array([[-0.000, -0.866,  0.500,  0.],
                             [ 1.000, -0.000, -0.000,  0.],
                             [ 0.000,  0.500,  0.866,  0.],
                             [ 0.000,  0.000,  0.000,  1.]]),
                   np.array([[ 0.866,  0.433, -0.250,  0.],
                             [-0.500,  0.750, -0.433,  0.],
                             [ 0.000,  0.500,  0.866,  0.],
                             [ 0.000,  0.000,  0.000,  1.]]),
                   np.array([[-0.866,  0.433, -0.250,  0.],
                             [-0.500, -0.750,  0.433,  0.],
                             [ 0.000,  0.500,  0.866,  0.],
                             [ 0.000,  0.000,  0.000,  1.]])]

    for pose in ligth_poses:
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0],
                                          intensity=1.0)
        scene.add(light, pose=pose)
    if interactive:
        pyrender.Viewer(scene, use_raymond_lighting=True)
    else:
        r = pyrender.OffscreenRenderer(*resolution)
        color, depth = r.render(scene)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.axis('off')
    
        ind_ax0 = np.where(np.any(np.any(color != 255, axis=2), axis=1))[0]
        ind_ax1 = np.where(np.any(np.any(color != 255, axis=2), axis=0))[0]
    
        ax.imshow(color[ind_ax0[0]:(ind_ax0[-1]+1), ind_ax1[0]:(ind_ax1[-1]+1), :])

        return ax
