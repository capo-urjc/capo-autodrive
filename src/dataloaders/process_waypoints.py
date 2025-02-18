import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


def get_speed(p1, p2, frames: int = 1, fps: int = 20):
    secs_per_frame = 1 / fps
    secs_total = secs_per_frame * frames
    distance = np.sqrt((p1[0, :] - p2[0, :])**2 + (p1[1, :] - p2[1, :])**2 + (p1[2, :] - p2[2, :])**2)
    return (distance / secs_total), (p2 - p1) / secs_total


# TODO: duplicate code, also in src/tools/visualization/routes.py
def get_waypoints(route_id):
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

def process_route(route_id: int):
    waypoints = get_waypoints(route_id)

    log_file = f"data/dataset/debug/Route{route_id}/ego_log.csv"
    data : pd.DataFrame = pd.read_csv(log_file)

    # fix bad files data, new files are ok
    if np.isnan(data.loc[0, 'v.z']):
        data['a.z'] = 0.0
        data['v.z'] = 0.0

    data['waypoint_id'] = np.nan
    data['waypoint.x'] = np.nan
    data['waypoint.y'] = np.nan
    data['waypoint.z'] = np.nan
    data['v.v'] = 0.0

    for i, waypoint in enumerate(waypoints):
        data['distance_to_wp'] = np.sqrt((data['t.x']-waypoint[0])**2 + (data['t.y']-waypoint[1])**2 + (data['t.z']-waypoint[2])**2)
        closest_idx = data['distance_to_wp'].idxmin()
        data.loc[:closest_idx, 'waypoint_id'] = data.loc[:closest_idx, 'waypoint_id'].fillna(i)
        data.loc[:closest_idx, 'waypoint.x'] = data.loc[:closest_idx, 'waypoint.x'].fillna(waypoint[0])
        data.loc[:closest_idx, 'waypoint.y'] = data.loc[:closest_idx, 'waypoint.y'].fillna(waypoint[1])
        data.loc[:closest_idx, 'waypoint.z'] = data.loc[:closest_idx, 'waypoint.z'].fillna(waypoint[2])

    data.drop(columns=['distance_to_wp'], inplace=True)
    data['waypoint_id'] = data['waypoint_id'].fillna(data.loc[i, 'waypoint_id'])
    data['waypoint_id'] = data['waypoint_id'].astype(int)
    data.loc[:closest_idx, 'waypoint.x'] = data.loc[:closest_idx, 'waypoint.x'].fillna(data.loc[:i, 'waypoint.x'])
    data.loc[:closest_idx, 'waypoint.y'] = data.loc[:closest_idx, 'waypoint.y'].fillna(data.loc[:i, 'waypoint.y'])
    data.loc[:closest_idx, 'waypoint.z'] = data.loc[:closest_idx, 'waypoint.z'].fillna(data.loc[:i, 'waypoint.z'])

    # calculate speed
    keypoints = data[['t.x', 't.y', 't.z']].to_numpy().T
    m_per_hour, vs = get_speed(keypoints[:, 0:-1], keypoints[:, 1:])

    data.loc[1:, 'v.x'] = vs[0, :]
    data.loc[1:, 'v.y'] = vs[1, :]
    data.loc[1:, 'v.z'] = vs[2, :]
    data.loc[1:, 'v.v'] = m_per_hour

    return data

for route_id in [0, 1, 2, 4, 5]:
    data = process_route(route_id)
    data.to_csv(f"data/dataset/debug/Route{route_id}/ego_log_waypoints.csv", index=False)

