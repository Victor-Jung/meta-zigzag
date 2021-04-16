from functools import reduce

import matplotlib.cm as cm
import numpy as np
import plotly.graph_objs as go
from scipy.spatial import Delaunay


def map_z2color(zval, colormap, vmin, vmax):
    # map the normalized value zval to a corresponding color in the colormap

    if vmin > vmax:
        raise ValueError('incorrect relation between vmin and vmax')
    t = (zval - vmin) / float((vmax - vmin))  # normalize val
    R, G, B, alpha = colormap(t)
    return 'rgb(' + '{:d}'.format(int(R * 255 + 0.5)) + ',' + '{:d}'.format(int(G * 255 + 0.5)) + \
           ',' + '{:d}'.format(int(B * 255 + 0.5)) + ')'


def tri_indices(simplices):
    # simplices is a numpy array defining the simplices of the triangularization
    # returns the lists of indices i, j, k

    return ([triplet[c] for triplet in simplices] for c in range(3))


def plotly_trisurf(x, y, z, simplices, colormap=cm.RdBu, plot_edges=None):
    # x, y, z are lists of coordinates of the triangle vertices
    # simplices are the simplices that define the triangularization;
    # simplices  is a numpy array of shape (no_triangles, 3)
    # insert here the  type check for input data

    points3D = np.vstack((x, y, z)).T
    tri_vertices = list(map(lambda index: points3D[index], simplices))  # vertices of the surface triangles
    zmean = [np.mean(tri[:, 2]) for tri in tri_vertices]  # mean values of z-coordinates of
    # triangle vertices
    min_zmean = np.min(zmean)
    max_zmean = np.max(zmean)
    facecolor = [map_z2color(zz, colormap, min_zmean, max_zmean) for zz in zmean]
    I, J, K = tri_indices(simplices)

    triangles = go.Mesh3d(x=x,
                          y=y,
                          z=z,
                          facecolor=facecolor,
                          i=I,
                          j=J,
                          k=K,
                          name=''
                          )

    if plot_edges is None:  # the triangle sides are not plotted
        return [triangles]
    else:
        # define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
        # None separates data corresponding to two consecutive triangles
        lists_coord = [[[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices] for c in range(3)]
        Xe, Ye, Ze = [reduce(lambda x, y: x + y, lists_coord[k]) for k in range(3)]

        # define the lines to be plotted
        lines = go.Scatter3d(x=Xe,
                             y=Ye,
                             z=Ze,
                             mode='lines',
                             line=dict(color='rgb(50,50,50)', width=1.5)
                             )
        return [triangles, lines]


def plot_dot(point_x, point_y, point_z):
    return go.Scatter3d(x=[point_x], y=[point_y], z=[point_z],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=z,  # set color to an array/list of desired values
                            colorscale='Viridis',  # choose a colorscale
                            opacity=1
                        )
                        )


def make_figure(x, y, z_array, point_x, point_y, point_z):
    # define 2D points, as input data for the Delaunay triangulation of U
    points2D = np.vstack([x, y]).T
    tri = Delaunay(points2D)  # triangulate the rectangle U
    figure = go.Figure(
        frames=[
            go.Frame(
                data=[
                    *plotly_trisurf(x, y, z_array[frame_id], tri.simplices, colormap=cm.RdBu, plot_edges=True),
                    plot_dot(point_x[frame_id], point_y[frame_id], point_z[frame_id])
                ],
                name=str(frame_id)
            )
            for frame_id in range(nb_frames)
        ]
    )
    figure.add_traces(figure.frames[0].data)
    return figure


def make_layout(figure):
    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    def add_sliders(layout):
        layout.sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(figure.frames)
                ],
            }
        ]
        return layout

    layout = go.Layout(
        title=f'Temporal mapping <br><span style="font-size: 7px;"></span>',
        scene=dict(
            xaxis_title='Loop idx',
            yaxis_title='Loop idy',
            zaxis_title='Utilization'),
        font=dict(
            family="Courier New, monospace",
            size=10,
            color="RebeccaPurple"
        ),
        scene_camera=dict(
            # center=dict(x=0, y=0, z=0.7),
            eye=dict(x=1.75, y=-1.5, z=1.25)
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ]
    )
    layout = add_sliders(layout)
    figure.update_layout(layout)
    return figure, layout


def test():
    x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
         4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7,
         7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14,
         14, 15, 15, 15, 15, 16, 16, 16, 17, 17, 18]
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
         16, 17, 18, 19, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
         15, 16, 17, 18, 19, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 9, 10,
         11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 11, 12, 13, 14, 15, 16, 17, 18, 19, 12,
         13, 14, 15, 16, 17, 18, 19, 13, 14, 15, 16, 17, 18, 19, 14, 15, 16, 17, 18, 19, 15, 16, 17, 18, 19, 16, 17, 18, 19,
         17, 18, 19, 18, 19, 19]
    z = [0.0, 0.0, 0.013191273465246053, -0.053589548452562155, 0.031973379629629636, 0.0, 0.006803542673107885, 0.0,
         -0.053589548452562155, 0.0, 0.031973379629629636, 0.031973379629629636, 0.006803542673107885, 0.0,
         0.06018518518518519, 0.031973379629629636, 0.031973379629629636, 0.0, 0.031973379629629636, 0.0,
         0.013191273465246053, -0.053589548452562155, 0.031973379629629636, 0.0, 0.006803542673107885, 0.0,
         -0.053589548452562155, 0.0, 0.031973379629629636, 0.031973379629629636, 0.006803542673107885, 0.0,
         0.06018518518518519, 0.031973379629629636, 0.031973379629629636, 0.0, 0.031973379629629636, 0.013191273465246053,
         -0.053589548452562155, 0.023011982570806097, 0.0, -0.0014679313459801113, 0.0, -0.053589548452562155, 0.0,
         0.023011982570806097, 0.023011982570806097, -0.0014679313459801113, 0.0, 0.043955472326258835,
         0.023011982570806097, 0.023011982570806097, 0.0, 0.023011982570806097, -0.04706209969367864, 0.0,
         -0.043295271556141116, -0.041952139807367686, -0.043295271556141116, -0.04706209969367864, -0.043295271556141116,
         0.0, 0.0, -0.041952139807367686, -0.043295271556141116, 0.03277099088617412, 0.0, 0.0, -0.043295271556141116, 0.0,
         0.05690235690235691, 0.010502112851106152, 0.01858449898829949, 0.020763525890184528, 0.0, 0.021878869751210187,
         0.20372935262131942, 0.20372935262131942, 0.022254578837268793, 0.021878869751210187, 0.2582704063286587,
         0.20372935262131942, 0.20372935262131942, 0.021878869751210187, 0.20372935262131942, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def generate_random_data(nb_frames):
        z_array = []
        point_x = []
        point_y = []
        point_z = []
        for frame_id in range(0, nb_frames):
            c_z = np.random.uniform(low=0, high=1, size=(len(z),))
            c_z = list(c_z)
            z_array.append(c_z)
            id = int(np.random.rand() * len(c_z))
            z_ = c_z[id]
            x_ = x[id]
            y_ = y[id]
            point_x.append(x_)
            point_y.append(y_)
            point_z.append(z_)
        return z_array, point_x, point_y, point_z


    nb_frames = 68
    z_array, point_x, point_y, point_z = generate_random_data(nb_frames)
    figure = make_figure(x, y, z_array, point_x, point_y, point_z)
    figure, layout = make_layout(figure)
    figure.show()
