import pyvista as pv
import numpy as np


## Load Mobius Dataset ##
mobius = pv.PolyData('mobius.vtk')
xyz = np.array(mobius.points)




##########################################
pv.set_plot_theme("document")
p = pv.Plotter()

p.disable_shadows()
silhouette = dict(
    color='black',
    line_width=6.,
    decimate=0.,
    feature_angle=True,
)
p.add_mesh(mobius, color='white', smooth_shading=True, silhouette=silhouette, ambient=0.7, opacity=1.)




# Compute the normals in-place and use them to warp the globe
mobius.compute_normals(inplace=True)  # this activates the normals as well
mobius.vectors = 0.1*np.array(mobius.point_normals)


ones = np.ones(mobius.points.shape[0])
mobius2 = mobius.warp_by_vector(factor= .01)
mobius3 = mobius.warp_by_vector(factor=-.01)


p.add_mesh(mobius.arrows,color='black', lighting=False,)

p.show()


