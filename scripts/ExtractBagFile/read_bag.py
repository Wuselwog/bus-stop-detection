# -*- coding: utf-8 -*-

"""
Extract images and GPS from a rosbag.
"""

import os
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image, NavSatFix
from cv_bridge import CvBridge
from exif import set_gps_location

def main():
    parser = argparse.ArgumentParser(description="Extract images and GPS from a rosbag.")
    parser.add_argument(
        "-i", "--input", default='./test.bag', help="Input ROS bag")
    parser.add_argument(
        "-c", "--cam-id", nargs='+', type=int, default=[3,], help="Selected camera IDs to extract")
    parser.add_argument(
        "-o", "--output", default='./output', help="Output dir")
    parser.add_argument(
        "-g", "--gps-save", action='store_true', help="Whether to save GPS as exif info of the images")
    args = parser.parse_args()

    bag_file = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    topics = ['/fix'] if args.gps_save else []
    for cam_id in args.cam_id:
        topics.append('/camera{}/image_raw/compressed'.format(cam_id))
    
    print("Extract images from {} for topics {}".format(bag_file, topics))

    bag = rosbag.Bag(bag_file, "r")
    # info_dict = yaml.load(bag._get_yaml_info())
    # print(info_dict)

    bridge = CvBridge()
    LAST_GPS = NavSatFix()
    print(LAST_GPS)
    count = 0
    for topic, msg, t in bag.read_messages(topics=topics):
        if 'image_raw' in topic:
            cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")

            time_stamps = '_{:0>10d}_{:0>9d}'.format(t.secs, t.nsecs)
            image_filename = topic[1:8] + time_stamps + '.jpg' 
            image_dir = os.path.join(output_dir, image_filename)
            
            cv2.imwrite(image_dir, cv_img)
            if args.gps_save:
                set_gps_location(image_dir, LAST_GPS.latitude, LAST_GPS.longitude, LAST_GPS.altitude)

        elif 'fix' in topic:
            LAST_GPS = msg

    bag.close()

    return

if __name__ == '__main__':
    main()