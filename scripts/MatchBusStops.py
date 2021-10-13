import os, cv2
import numpy as np

import piexif
from PIL import Image

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from sys import maxsize

def GetStopDict():
    with open("../data/stops.txt",'r') as f:
        stop_data = f.read()
    stops = stop_data.split("\n")
    keys = stops[0].split(",")
    return [dict(zip(keys,s.split(","))) for s in stops[1:-1]]

def DMS_to_DD(exif_dict):
    lat = [float(x)/float(y) for x, y in exif_dict['GPS'][2]]
    latref = exif_dict['GPS'][1]
    lon = [float(x)/float(y) for x, y in exif_dict['GPS'][4]]
    lonref = exif_dict['GPS'][3]

    lat = lat[0] + lat[1]/60 + lat[2]/3600
    lon = lon[0] + lon[1]/60 + lon[2]/3600
    if latref == b'S':
        lat = -lat
    if lonref == b'W':
        lon = -lon
    return [lat, lon]

def deg_to_meter_dist(deg):
    return deg * 1.11 * 100000.

def save_img(directory, img_name, im, exif_bytes):
    os.makedirs(directory, exist_ok=True)
    image_dir = os.path.join(directory, img_name)
    cv2.imwrite(image_dir, im)
    piexif.insert(exif_bytes, image_dir)

def MatchStops(predictedStops, output_dir, output_dir_officials, output_dir_unofficials, print_output=False):
    stops_dict = GetStopDict()
    for stop in predictedStops:
        if print_output:
            print("Stop:", stop)
        img_name = stop.split('/')[-1]
        image_dir = os.path.join(output_dir, img_name)
        im = cv2.imread(stop)
        
        pil_img = Image.open(stop)
        exif_dict = piexif.load(pil_img.info["exif"])
        exif_bytes = piexif.dump(exif_dict)
        
        pred_stop_loc = np.array(DMS_to_DD(exif_dict))
        min_dist = maxsize
        nearest_stop = None
        for s in stops_dict:
            stop_loc = np.array([float(s["stop_lat"]), float(s["stop_lon"])])
            dist = np.linalg.norm(stop_loc - pred_stop_loc)
            dist = deg_to_meter_dist(dist)
            if dist < min_dist:
                min_dist = dist
                nearest_stop = s

        if print_output:        
            print(min_dist, " - ", stop, " - ", nearest_stop["stop_name"], " - ", pred_stop_loc, " - ", [float(nearest_stop["stop_lat"]), float(nearest_stop["stop_lon"])])
        
        save_img(output_dir, img_name, im, exif_bytes)
        if min_dist < 100:
            save_img(output_dir_officials, img_name, im, exif_bytes)
        else:
            save_img(output_dir_unofficials, img_name, im, exif_bytes)