# -*- coding: utf-8 -*-

"""
Extract images and GPS from a rosbag.
"""

import os
from os.path import isfile, join
import argparse

import cv2

import rosbag, rospy
from sensor_msgs.msg import Image, NavSatFix
from cv_bridge import CvBridge
from exif import set_gps_location

def write_images(img_buffer, args):
    for img in img_buffer:
        image_dir, cv_img, LAST_GPS = img
        # img_buffer.append(image_dir, cv_img, LAST_GPS)
        cv2.imwrite(image_dir, cv_img)
        if args.gps_save:
            set_gps_location(image_dir, LAST_GPS.latitude, LAST_GPS.longitude, LAST_GPS.altitude)

def main():
    # latitude, longitude and width in degrees of break areas   
    washington_depot_loc = [40.224142, -80.216757, 90. / 1.11 / 100000.]
    pittsburgh_pause_loc = [40.446020, -79.988753, 90. / 1.11 / 100000.]
    gas_station_loc = [40.17822, -80.26139, 50. / 1.11 / 100000.]
    # washington_pause_loc = [40.172611, -80.244531]
    pause_locs = [washington_depot_loc, pittsburgh_pause_loc, gas_station_loc]

    parser = argparse.ArgumentParser(description="Extract images and GPS from a rosbag.")
    parser.add_argument(
        "-f", "--folder", default='.', help="The folder from which all Ros Bags should get read")
    parser.add_argument(
        "-i", "--input", nargs='+', type=str, default=[], help="Input ROS bags")
    #parser.add_argument(
    #    "-i", "--input", default='./test.bag', help="Input ROS bag")
    parser.add_argument(
        "-c", "--cam-id", nargs='+', type=int, default=[3,], help="Selected camera IDs to extract")
    parser.add_argument(
        "-o", "--output", default='./output', help="Output dir")
    parser.add_argument(
        "-g", "--gps-save", action='store_true', help="Whether to save GPS as exif info of the images")
    parser.add_argument(
        "-t", "--time", nargs='+', type=int, default=[0, ], help="Selected time to extract n frames before")
    parser.add_argument(
        "-n", "--num_images", type=int, default=0, help="Amount of frames that should be extracted")
    # parser.add_argument(
        # "-r", "--recurse", action='store_true', help="Extra")
    args = parser.parse_args()

    bag_files = args.input
    folder = args.folder
    output_dir = args.output
    frames = args.time
    num_images = args.num_images

    extract(bag_files, output_dir, folder, frames, num_images, args.gps_save, args.cam_id)

def extract(bag_files, output_dir, folder, frames, num_images, gps_save, cam_id):
    os.makedirs(output_dir, exist_ok=True)

    topics = ['/fix'] if gps_save else []
    # topics.append('/velocity')
    for cam_id in cam_id:
        topics.append('/camera{}/image_raw/compressed'.format(cam_id))

    if len(bag_files) == 0:
        bag_files = imgs = sorted([join(folder, f) for f in os.listdir(folder) if isfile(join(folder, f)) and f[-4:] == ".bag"])

    bridge = CvBridge()

    bus_stopped = False
    img_buffer = []
    velocity_threshold = 4

    frame_idx = 0
    current_frame = frames[0]
    print("Looking for img ", current_frame)
    found_image = False
    finished = False

    for num, bag_file in enumerate(bag_files):
        print(num, " / ", len(bag_files))
        print("Extract images from {} for topics {}".format(bag_file, topics))

        bag = rosbag.Bag(bag_file, "r")
        # info_dict = yaml.load(bag._get_yaml_info())
        # print(info_dict)

        found_image = True
        while (found_image):
            found_image = False

            if gps_save:
                LAST_GPS = NavSatFix()
                print(LAST_GPS)

            velocity = 0
            for topic, msg, t in bag.read_messages(topics=topics, start_time=rospy.Time(current_frame - num_images), end_time=rospy.Time(current_frame + num_images)):
                if 'velocity' in topic:
                    print(velocity)
                #     velocity = msg.velocity
                #     if velocity <= 0.2:
                #         bus_stopped = True
                #     elif velocity > velocity_threshold:
                #         if bus_stopped:
                #             write_images(img_buffer, args)
                #         img_buffer.clear()
                #         bus_stopped = False
                        
                elif 'image_raw' in topic:
                    # Check if the bus is currently doing a break
                    # if abs(t.secs - frame) > num_images or t.secs > frame:
                        # continue

                    cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")

                    time_stamps = '_{:0>10d}_{:0>9d}'.format(t.secs, t.nsecs)
                    image_filename = topic[1:8] + time_stamps + '.jpg' 
                    image_dir = os.path.join(output_dir, image_filename)
                    
                    # img_buffer.append((image_dir, cv_img, LAST_GPS))
                    cv2.imwrite(image_dir, cv_img)
                    if gps_save:
                        set_gps_location(image_dir, LAST_GPS.latitude, LAST_GPS.longitude, LAST_GPS.altitude)
                    
                    if not found_image and frame_idx + 1 < len(frames): 
                        frame_idx += 1
                        found_image = True
                        current_frame = frames[frame_idx]
                        print("Looking next for img ", current_frame)
                    elif not found_image and not finished:
                        print("Found all images")
                        finished = True

                elif 'fix' in topic:
                    LAST_GPS = msg
                    # print(LAST_GPS)

            if finished:
                bag.close()
                return

        bag.close()
    # if bus_stopped:
        # write_images(img_buffer, args)

    return

if __name__ == '__main__':
    main()
