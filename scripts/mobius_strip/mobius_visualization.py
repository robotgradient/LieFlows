# sphinx_gallery_thumbnail_number = 12

import numpy as np
import pyvista as pv
from math import pi
import matplotlib.pyplot as plt

pv.set_plot_theme("document")
# p = pv.Plotter()
p = pv.Plotter()

mobius = pv.ParametricMobius()
mobius.point_arrays['scalars'] = np.random.rand(mobius.n_points)

p.disable_shadows()
silhouette = dict(
    color='black',
    line_width=6.,
    decimate=0.,
    feature_angle=True,
)
p.add_mesh(mobius, smooth_shading=True, silhouette=silhouette, ambient=0.7, opacity=1.)


p.show()