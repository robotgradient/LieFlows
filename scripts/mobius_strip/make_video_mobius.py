import os
import numpy as np
import pyvista as pv
from liesvf.riemannian_manifolds.riemannian_manifolds import Mobius
from liesvf.visualization.mobius_visualization import visualize_vector_field, visualize_latent_points_on_mobius

## Create simple stable dynamics
def svf(x):
    ## Stable Vector Field ##
    dx = -10.1*x

    ## Limit Cycle Field ##
    dx[:,0] = np.ones_like(dx[:,0])*10.
    dx[:,1] = np.zeros_like(dx[:,0])

    #dx[:,1] = np.zeros_like(dx[:,1])
    norm = np.sqrt(np.sum(dx**2,axis=1))

    thrs = 0.05
    norm_thrs = norm>thrs
    norm = norm_thrs*norm/thrs + (1- norm_thrs)*np.ones_like(norm)
    norm = np.tile(norm,(2,1)).T
    return dx/norm




## Load Mobius Dataset ##
mobius = pv.PolyData('mobius.vtk')
xyz = np.array(mobius.points)

## Test Mobius Map works well xyz -> uv -> xyz
mobius_map = Mobius()




##########################################


def gen_visual(off_screen=False, roll = 180):
    pv.set_plot_theme("document")

    p = pv.Plotter(off_screen=off_screen, window_size=[2024, 2024])

    p.disable_shadows()
    silhouette = dict(
        color='black',
        line_width=6.,
        decimate=None,
        feature_angle=True,
    )
    p.add_mesh(mobius, color='white', smooth_shading=True, silhouette=silhouette,  ambient=0.7, opacity=1.)

    p.camera_position = 'yz'
    #p.camera.position = (0., 0., 0.)
    p.camera.roll += roll
    p.camera.azimuth = 45
    p.camera.elevation += 90


    visualize_vector_field(p, mobius, svf)
    return p



dir = os.path.abspath(os.path.dirname(__file__))
save_dir = os.path.join(dir,'video')


x0 = np.random.rand(50,2)
x0[:,0] = x0[:,0]*2*np.pi - np.pi
x0[:,1] = x0[:,1]*2 -1

T=5000
dt = 0.1

#p = gen_visual(off_screen=False)
roll = 0
d_roll = 0.3
for t in range(T):
    print('iter: {}'.format(t))
    v = svf(x0)
    x1 = x0+ v*dt
    x0 = x1
    #x0[:,0] = np.arctan2(np.sin(x0[:,0]), np.cos(x0[:,0]))

    roll +=d_roll
    p = gen_visual(off_screen=True, roll=roll)

    visualize_latent_points_on_mobius(p, mobius, x0)

    save_fig = 'mobius_test_{}.png'.format(t)
    save_file = os.path.join(save_dir, save_fig)

    p.show(screenshot=save_file)
    p.close()
    p.deep_clean()
#p.show()

## Make Video From figures ##




