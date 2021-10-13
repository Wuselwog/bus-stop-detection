# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-

"""
Extract images and GPS by playing a rosbag and subscribing to the topics.
"""

import argparse
import os

import cv2
import rospy

# Don't need to build cv_bridge using python3 if we only use compressed_imgmsg_to_cv2()
from cv_bridge import CvBridge
from exif import set_gps_location
from sensor_msgs.msg import CompressedImage, NavSatFix

# Your own model
# from your_filter import YourFilter
from Detection import Tracker, GetPredictor

CUR_GPS = NavSatFix()
CUR_VEL = 0

def main(args):
    camera_name = "camera" + str(args.cam_id)
    output_dir = args.output
    save_gps = args.save_gps
    os.makedirs(output_dir, exist_ok=True)

    rospy.init_node(camera_name + "_filter_node")
    rospy.loginfo("Initialized filter node for " + camera_name)

    # Your own model
    # model_dir = "path_to_saved_model"
    # model = YourFilter(model_dir)
    print("Init tracker")
    tracker = Tracker()
    tracker.Init()
    predictor = GetPredictor()

    image_sub = rospy.Subscriber(
        camera_name + "/image_raw/compressed",
        CompressedImage,
        img_callback,
        # callback_args=(model),
        callback_args=(output_dir, save_gps, tracker, predictor),
        queue_size=100,
        buff_size=2 ** 24,
    )
    gps_sub = rospy.Subscriber("/fix", NavSatFix, gps_callback, queue_size=100)
    # gps_sub = rospy.Subscriber("/velocity", Vel, velocity_callback, queue_size=1)
    # spin() simply keeps python from exiting until this node is stopped
    print("Finished setup")
    rospy.spin()

def velocity_callback(msg, args):
    print(msg, " and ", args)
    global CUR_VEL
    CUR_VEL = 0

def img_callback(msg, args):
    global CUR_GPS
    global CUR_VEL

    # ignore images if the bus currently has a high speed
    if CUR_VEL > 4:
        return

    # latitude, longitude and width in degrees of break areas   
    washington_depot_loc = [40.224142, -80.216757, 90. / 1.11 / 100000.]
    pittsburgh_pause_loc = [40.446020, -79.988753, 90. / 1.11 / 100000.]
    gas_station_loc = [40.17822, -80.26139, 50. / 1.11 / 100000.]
    # washington_pause_loc = [40.172611, -80.244531]
    pause_locs = [washington_depot_loc, pittsburgh_pause_loc, gas_station_loc]

    # Check if the bus is currently doing a break
    for loc in pause_locs:
        # print(abs(LAST_GPS.latitude - loc[0]), " and",  abs(LAST_GPS.longitude - loc[1]))
        if abs(CUR_GPS.latitude - loc[0]) <= loc[2] and abs(CUR_GPS.longitude - loc[1]) <= loc[2]:
            return

    # model = args[0]
    output_dir = args[0]
    save_gps = args[1]
    tracker = args[2]
    predictor = args[3]
    bridge = CvBridge()
    frame = bridge.compressed_imgmsg_to_cv2(
        msg, desired_encoding="passthrough"
    )  # BGR images

    # Your codes here to process Image data
    # frame = frame[:, :, ::-1]  # BGR to RGB
    # output = model(frame)


# cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")

#                 time_stamps = '_{:0>10d}_{:0>9d}'.format(t.secs, t.nsecs)
#                 image_filename = topic[1:8] + time_stamps + '.jpg' 
#                 image_dir = os.path.join(output_dir, image_filename)

    # print("MSG:", msg.header.frame_id[5])

    # This is only for image extraction
    t = msg.header.stamp
    time_stamps = "_{:0>10d}_{:0>9d}".format(t.secs, t.nsecs)
    camera_num = msg.header.frame_id[5]
    image_filename = "camera" + camera_num + time_stamps + ".jpg"
    image_dir = os.path.join(output_dir, image_filename)
    
    tracker.TrackData(image_filename, frame, None, predictor=predictor, view="", slideshow=False, deleteAfter=1, tracking_overlap_threshold=0.0, atLeastSeenFor=1)

    # cv2.imwrite(image_dir, frame)
    # if save_gps:
    #     set_gps_location(
    #         image_dir, CUR_GPS.latitude, CUR_GPS.longitude, CUR_GPS.altitude
    #     )


def gps_callback(data):
    # Your codes here to process GPS data
    global CUR_GPS
    CUR_GPS = data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform tasks in the bus stop detection pipeline."
    )
    parser.add_argument(
        "-c",
        "--cam-id",
        type=int,
        default=3,
        help="Select camera ID to extract",
    )
    parser.add_argument("-o", "--output", default="./output", help="Output dir")
    parser.add_argument(
        "-g",
        "--save-gps",
        action="store_true",
        help="Whether to save GPS as exif info of the images",
    )
    args = parser.parse_args()
    main(args)
