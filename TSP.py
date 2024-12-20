import numpy as np
#from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.heuristics import solve_tsp_local_search
import geopandas as gpd
from shapely import Point



def hoverPointsTSP(points, scramble=False):
    # print(f'points: {points}')
    if len(points) == 0:
        gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in points], crs='EPSG:3857')
        return gdf, 0
    depotLocation = (0, 0)
    points = np.insert(points, 0, depotLocation, axis=0)
    distanceMatrix = euclidean_distance_matrix(points)
    if scramble == False:
        initPerm = list(range(len(points)))
        permutation, distance = solve_tsp_local_search(distance_matrix=distanceMatrix, x0=initPerm)
    else:
        permutation, distance = solve_tsp_local_search(distance_matrix=distanceMatrix)
    orderedPoints = [points[i] for i in permutation if i != 0]
    gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in orderedPoints], crs='EPSG:3857')
    return gdf, distance