import os
import torch
import numpy as np
import torch.nn as nn
import plotly.graph_objects as go
from config.config import settings
import src.optimizer as optimizer


def pathological_curve_loss(w1, w2):
    x1_loss = torch.tanh(w1)**2 + 0.01 * torch.abs(w1)
    x2_loss = torch.sigmoid(w2)
    return x1_loss + x2_loss

def plot_curve(curve_function, x_range=(-5,5), y_range=(-5,5),
               title="Pathological curvature"):
    x = torch.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / 100.)
    y = torch.arange(y_range[0], y_range[1], (y_range[1] - y_range[0]) / 100.)
    x, y = torch.meshgrid(x, y, indexing='xy')
    z = curve_function(x, y)
    x, y, z = x.numpy(), y.numpy(), z.numpy()

    fig = go.Figure()
    fig.add_trace(go.Surface(x=x, y=y, z=z,colorscale="Viridis",
                             showscale=False,
                             contours={"z": {"show": True, "usecolormap": True,
                                             "project_z": True}
                                       }
                             )
                  )
    fig.update_layout(title=f"{title}",
                      scene =dict(xaxis=dict(title=f"w_1"),
                                  yaxis=dict(title=f"w_2"),
                                  zaxis=dict(title=f"Loss")
                                  )
                      )

    fig.write_html(os.path.join(settings.artifacts, f"{title}.html"))
    return fig

def train_curve(optimizer_func, curve_func=pathological_curve_loss, num_updates=100, init=[-1,-1]):
    weights = nn.Parameter(torch.FloatTensor(init), requires_grad=True)
    optimizer = optimizer_func([weights])

    list_points = []
    for i in range(num_updates):
        loss = curve_func(weights[0], weights[1])
        list_points.append(torch.cat([weights.data.detach(), loss.unsqueeze(dim=0).detach()], dim=0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        points = torch.stack(list_points, dim=0).numpy()
    return points

def plot_weights():

    SGD_points = train_curve(lambda params: optimizer.SGD(params, lr=10))
    SGDMom_points = train_curve(lambda params: optimizer.SGDMomentum(params, lr=10, momentum=0.9))
    Adam_points = train_curve(lambda params: optimizer.Adam(params, lr=1))
    all_points = np.concatenate([SGD_points, SGDMom_points, Adam_points], axis=0)

    figure = plot_curve(pathological_curve_loss,
                    x_range=(-np.absolute(all_points[:, 0]).max(), np.absolute(all_points[:, 0]).max()),
                    y_range=(all_points[:, 1].min(), all_points[:, 1].max()))

    figure.add_trace(go.Scatter3d(x=SGD_points[:, 0], y=SGD_points[:, 1],z=SGD_points[:, 2],
                                name="SGD"))
    figure.add_trace(go.Scatter3d(x=SGDMom_points[:, 0], y=SGDMom_points[:, 1],z=SGDMom_points[:, 2],
                                  name="SGDMom"))
    figure.add_trace(go.Scatter3d(x=Adam_points[:, 0], y=Adam_points[:, 1],z=Adam_points[:, 2],
                                name="Adam"))
    figure.write_html(os.path.join(settings.artifacts, f"PathologicalCurve_TrainingPlot.html"))
