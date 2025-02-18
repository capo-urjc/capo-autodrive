import colorsys
import xml.etree.ElementTree as ET
import plotly.graph_objects as go

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def get_waypoints(route_id):
    log_file = f"data/dataset/debug/Route{route_id}/ego_log_waypoints.csv"
    data = pd.read_csv(log_file)
    return data[['t.x','t.y','t.z']].to_numpy(), data[['waypoint']].to_numpy()

def get_keypoints(route_id):
    routes_file = "/raid/code/CARLA_Leaderboard/v2.0/leaderboard/data/routes_training.xml"
    tree = ET.parse(routes_file)
    root = tree.getroot()

    waypoints = []
    route = root.find(f"./route[@id='{route_id}']")
    if route:  # Match the given route ID
        for waypoint in route.find("waypoints").findall("position"):
            x, y, z = float(waypoint.get("x")), float(waypoint.get("y")), float(waypoint.get("z"))
            waypoints.append([x, y, z])
    else:
        raise Exception(f"Route with id {route_id} not found")
    return np.array(waypoints)


def generate_colors(n, cmap_name="viridis"):
    cmap = plt.get_cmap(cmap_name)
    # return [f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a:.2f})"
    #         for r, g, b, a in [cmap(i / max(1, n - 1)) for i in range(n)]]
    return [f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 1.0)"
                for i in range(n)
                for r, g, b in [colorsys.hsv_to_rgb(i / n, 1, 1)]]

route_id = 5
keypoints, waypoint = get_waypoints(route_id)
keypoints = keypoints.T
wx = keypoints[0]
wy = keypoints[1]
wz = keypoints[2]

waypoints = get_keypoints(route_id).T
kx = waypoints[0]
ky = waypoints[1]
kz = waypoints[2]

colors = np.array(generate_colors(len(kx)))
keypoints_colors = colors[waypoint.flatten()]

fig = go.Figure(data=[go.Scatter(
    x=kx,
    y=ky,
    # mode='markers',
    marker=dict(size=20, color=list(colors), opacity=0.8),
    mode='lines+markers',
    line=dict(width=2, color='black')
)])

fig.add_trace(go.Scatter(
    x=wx,
    y=wy,
    mode='markers',
    marker=dict(size=5, color=keypoints_colors, opacity=0.8),
    # mode='lines+markers',
    # line=dict(width=2, color='blue')
))

fig.update_layout(
    title=f"Route {route_id} Waypoints",
    scene=dict(
        xaxis=dict(title="X Coordinate"),
        yaxis=dict(title="Y Coordinate"),
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.write_html(f"output/route{route_id}.html")
