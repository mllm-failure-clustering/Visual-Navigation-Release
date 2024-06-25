import sys, os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # add Visual-Navigation-Release to PYTHONPATH
os.environ['PYOPENGL_PLATFORM'] = 'egl' # we have to explicitly choose the opengl platform on linux

import numpy as np
from matplotlib import pyplot as plt

import sbpd.sbpd_renderer as sbpdr
import params.renderer_params as rp

import params.simulator.sbpd_simulator_params as sp
import simulators.sbpd_simulator as sbpdsim


def render_images(pos_nk3):
    params = rp.create_params()
    renderer = sbpdr.SBPDRenderer.get_renderer(params)
    starts_n2 = pos_nk3[:, :2]
    theta_n1 = pos_nk3[:, 2].reshape(-1, 1)
    imgs = renderer.render_images(starts_n2=starts_n2, thetas_n1=theta_n1)
    imgs = imgs.astype(int)
    return imgs

def render_images_with_sim(pos_nk3):
    params = sp.create_params()
    simulator = sbpdsim.SBPDSimulator(params)
    imgs = simulator.get_observation(pos_n3=pos_nk3)
    return imgs


# x, y, theta of the robot. e.g.
# pos_nk3 = np.array([[0, 0, 0],
#                     [0, 10, 0],
#                     [0, 20, 0]])

pos_nk3 = np.array([[420, 420, -90]])

imgs = render_images(pos_nk3)
# imgs = render_images_with_sim(pos_nk3)
imgconcat = np.concatenate(imgs, axis=1)
plt.imshow(imgconcat)
plt.savefig('/home/leo/git/Visual-Navigation-Release/sbpd/test1.jpg')