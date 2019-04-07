import sys
import numpy as np
import matplotlib.pyplot as plt
import dcm2npy
from PIL import Image
import tools
import re

def save_wrong_img(log_path):
    with open(log_path, 'r') as log:
        log_content = log.read()
        nodules = tools.get_nodules()

        pattern = re.compile('\d+(?=[^/]+.npy)')
        for nodule_id in pattern.findall(log_content):
            dcm2npy.save_img(tools.data_path, nodules[int(nodule_id)], (10, 20, 20))


def plot_npy(npy_file):
    '''
       plot the cubic slice by slice

    :param npy_file:
    :return:
    '''
    cubic_array = np.load(npy_file)
    # cubic_array = Image.open(npy_path)
    for i in range(cubic_array.shape[0]):
        plt.title(npy_path)
        plt.axis('on')
        plt.imshow(cubic_array[i,:,:])
        plt.show()

def plot_3d_npy(image):
    '''
        plot the 3D cubic
    :param image:   image saved as npy file path
    :return:
    '''
    from skimage import measure, morphology
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    image = np.load(image)
    verts, faces = measure.marching_cubes(image,0)
    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])
    plt.show()


if __name__ == '__main__':
    save_wrong_img('log/wrong.txt')
