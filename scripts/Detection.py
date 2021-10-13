import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
import random

import json

from IPython.display import clear_output
from time import sleep

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog, DatasetCatalog

from Data import GetCocoClasses

def GetPredictor():
    cfg = get_cfg()

    model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    predictor = DefaultPredictor(cfg)
    return predictor

# https://stackoverflow.com/questions/59076054/effificient-distance-like-matrix-computation-manual-metric-function
def compute_iou(bbox_a, bbox_b):
    xA = max(bbox_a[0], bbox_b[0])
    yA = max(bbox_a[1], bbox_b[1])
    xB = min(bbox_a[2], bbox_b[2])
    yB = min(bbox_a[3], bbox_b[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    boxBArea = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def IoU_door(box):
    return compute_iou(box, [45, 250, 95, 385])

def intersects_with_mirror(box):
    return compute_iou(box, [58, 166.8, 72, 211.8]) >= 0.5

def intersects_with_door(box):
    # Somebody who exited was at (64.35417175292969, 208.9422607421875)
    return IoU_door(box) >= 0.3

# TODO: update data file for this frame
def update_people_count(humans_to_count, data, atLeastSeenFor, img_name):
    actionFlag = False
    for idx in humans_to_count:
        human = humans_to_count[idx]
        if (not human["wasCounted"]) and human["seenFor"] >= atLeastSeenFor:
            is_at_door = intersects_with_door(human["box"])
            if is_at_door != human["firstSeenNearDoor"]:
                actionFlag = True
                if img_name not in data:
                    data[img_name] = {}
                if is_at_door:
                    data["Entered"] += 1
                    print("+1 entered - ", idx, " img ", img_name)
                    data[img_name]["Entered"] = data[img_name].get("Entered", 1)
                else:
                    data["Left"] += 1
                    print("+1 left - ", idx, " img ", img_name)
                    data[img_name]["Left"] = data[img_name].get("Left", 1)
                human["wasCounted"] = img_name
    return data, actionFlag

def final_update_people_count(humans_to_count, data, atLeastSeenFor, img_name):
    actionFlag = False
    for idx in humans_to_count:
        human = humans_to_count[idx]
        if (not human["wasCounted"]) and human["seenFor"] >= atLeastSeenFor and intersects_with_door(human["box"]):
            data["Unsure"] += 1
            print("+1 unsure - ", idx, " img ", img_name)
            actionFlag = True
            if img_name not in data:
                data[img_name] = {}
            data[img_name]["Unsure"] = data[img_name].get("Unsure", 1)
            human["wasCounted"] = img_name
        # human was counted as leaving, but in the end he was seen again near the door. This is often the case for people coming from in front of the bus
        # as it is hard to decide these cases, we note them as Unsure, e.g. so that a human could decide them
        elif human["wasCounted"] and human["firstSeenNearDoor"] and intersects_with_door(human["box"]):
            data[human["wasCounted"]]["Left"] -= 1
            data[human["wasCounted"]]["Unsure"] = data[human["wasCounted"]].get("Unsure", 1)
            print("+1 unsure -1 Left - ", idx, " img ", img_name)
            data["Left"] -= 1
            data["Unsure"] += 1
            
    return data, actionFlag

def ViewImg(img_name, im, humans, old_humans, view, metadata):
    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    

    fig = plt.figure(figsize=(128, 72), dpi=10)
    axe = plt.gca()
    axe.set_title(img_name)
    
    out = v.draw_instance_predictions(humans.to("cpu"))
    plt.imshow(out.get_image()[:, :, ::1])

    # draw in the check area for the door
    # 50 < position[0] < 80 and 190 < position[1] < 230
    # r_x = 45 * 635 / 1280
    # r_y = 250 * 355 / 720
    # r_w = 50 * 635 / 1280
    # r_h = 135 * 355 / 720
    # rect = Rectangle((r_x, r_y), r_w, r_h, linewidth=10, edgecolor='r', facecolor='none')
    # axe.add_patch(rect)
    
    # draw in the area for the mirror
    # r_x = 58 * 635 / 1280
    # r_y = 166.8 * 355 / 720
    # r_w = 14 * 635 / 1280
    # r_h = 45 * 355 / 720
    # rect = Rectangle((r_x, r_y), r_w, r_h, linewidth=10, edgecolor='r', facecolor='none')
    # axe.add_patch(rect)

    # draw the boxes where the humans were tracked in the last frame
    # old_boxes = [x["box"] for x in self.old_humans.values()]
    # for b in old_boxes:
    #     rect = Rectangle((b[0] * 635 / 1280, b[1] * 355 / 720), (b[2] - b[0]) * 635 / 1280, (b[3] - b[1]) * 355 / 720, linewidth=10, edgecolor='r', facecolor='none')
    #     plt.gca().add_patch(rect)

    # Add the patch to the Axes
#         bbox = plt.gca().get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    if view == "near":
        print("img: ", img_name)


class Tracker:
    def Init(self):
        self.tracked_humans = {}
        self.old_ids = []
        self.instance_count = -1
        self.class_names = GetCocoClasses()
        
        self.data = {}
        if "Entered" not in self.data:
            self.data["Entered"] = 0
        if "Left" not in self.data:
            self.data["Left"] = 0
        if "Unsure" not in self.data:
            self.data["Unsure"] = 0

    def DetectInDataset(self, dataset_dicts, metadata, predictor=None, num_imgs=10, start=0, step=1, sample_random=False, view="humans", slideshow=False, deleteAfter=2, tracking_overlap_threshold=0.0, atLeastSeenFor=1):
        if sample_random:
            selected_data = random.sample(dataset_dicts, num_imgs)
        else:
            selected_data = dataset_dicts[start:start + step * num_imgs:step]
            
        for d in selected_data:
            img_name = d["file_name"]
            im = cv2.imread(img_name)
            self.TrackData(img_name, im, metadata, predictor, view, slideshow, deleteAfter, tracking_overlap_threshold, atLeastSeenFor)

        if len(selected_data) > 0:
            self.OnDetectionFinished(atLeastSeenFor, img_name, im, view, metadata)
        else:
            print("Warning: Dataset is empty!")
        return self.data
        

    def TrackData(self, img_name, im, metadata, predictor=None, view="humans", slideshow=False, deleteAfter=1, tracking_overlap_threshold=0.0, atLeastSeenFor=1):
        # perform histogram equalization (seems to make no difference)
#         img_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
#         # equalize the histogram of the Y channel
#         img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
#         # convert the YUV image back to RGB format
#         im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        # remove the pixels on the left side, as the bus windows can confuse the algorithm and yield no benefits
        im[:, :40] = np.zeros(np.shape(im[:, :40]))
        
        self.old_humans = self.tracked_humans.copy()
        
        # increase the "not seen" time of each human and delete long unseen humans
        to_delete = {k: v for k, v in self.tracked_humans.items() if v["unseenFor"] > deleteAfter}
        self.data, actionFlag = final_update_people_count(to_delete, self.data, atLeastSeenFor, img_name)
        if view == "near" and actionFlag:
            ViewImg(img_name, im, self.humans, self.old_humans, view, metadata)
                
        # delete all unseen people after some time
        self.tracked_humans = {k: v for k, v in self.tracked_humans.items() if v["unseenFor"] <= deleteAfter}
        
        for idx in self.tracked_humans:
            self.tracked_humans[str(idx)]["unseenFor"] += 1
#             self.tracked_humans[str(idx)]["seenFor"] += 1

        visualizer = Visualizer(im[:, :, ::-1],
                   metadata=metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        
        if predictor is None:
            out = visualizer.draw_dataset_dict(d)
        else:
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            predictions = outputs["instances"].to("cpu")
            predicted_classes = predictions.get('pred_classes').to("cpu")
            
            # Store all detections
            for i, class_num in enumerate(predicted_classes):
                if img_name not in self.data:
                    self.data[img_name] = {}
                class_name = self.class_names[class_num.item() + 1]
                if class_name not in self.data[img_name]:
                    self.data[img_name][class_name] = []
                self.data[img_name][class_name].append({"Box": str(predictions.get('pred_boxes')[i])})
            
            human_ids = np.where(predictions.get('pred_classes').to("cpu").numpy() == 0)
            self.humans = predictions[human_ids]
    
            np_boxes = [b.numpy() for b in self.humans.get('pred_boxes')]
            new_human_poses = [((b[2] + b[0]) / 2., (b[3] + b[1]) / 2.) for b in np_boxes]
            
            # check that it is not the misdetected mirror
            delete_ids = np.where(predictions.get('pred_classes').to("cpu").numpy() == 0)
            
            keep_ids = []
            for i in range(len(np_boxes)):
                keep_ids.append(not intersects_with_mirror(np_boxes[i]))
            self.humans = self.humans[keep_ids]
            np_boxes = np.array(np_boxes)
            np_boxes = np_boxes[keep_ids]
            new_human_poses = np.array(new_human_poses)
            new_human_poses = new_human_poses[keep_ids]
            
            if len(self.humans) > 0 and len(self.tracked_humans) > 0:
                old_poses = [x["box"] for x in self.tracked_humans.values()]
                # calculate the distances for all matchings
                C = cdist(np_boxes, old_poses, lambda u, v: compute_iou(u, v))

                if len(self.humans) - len(self.tracked_humans) > 0:
                    padding = np.zeros((len(C) , len(self.humans) - len(self.tracked_humans)))
                    C = np.hstack((C, padding))
                
                # Get the best matching
                row_ind, assignment = linear_sum_assignment(C, maximize=True)
                cost = C[row_ind, assignment].sum()
                
                new_ids = []
                num_tracked = len(self.tracked_humans)
                for j in range(len(self.humans)):
                    i = assignment[j]
                    if i < num_tracked and compute_iou(np.array(list(self.tracked_humans.values())[i]["box"]), np.array(np_boxes[j])) > tracking_overlap_threshold:
                        idx = list(self.tracked_humans.keys())[i]
                        new_ids.append(int(idx))
                        self.tracked_humans[str(idx)]["unseenFor"] = 0
                        self.tracked_humans[str(idx)]["seenFor"] += 1
                        self.tracked_humans[str(idx)]["midpoint"] = new_human_poses[j]
                        self.tracked_humans[str(idx)]["box"] = np_boxes[j]
                        if self.tracked_humans[str(idx)]["seenFor"] == 1 and not self.tracked_humans[str(idx)]["firstSeenNearDoor"]:
                            self.tracked_humans[str(idx)]["firstSeenNearDoor"] = intersects_with_door(np_boxes[j])
                    else:
                        self.instance_count+=1
                        new_ids.append(self.instance_count)
                        self.tracked_humans[str(self.instance_count)] = {
                            "midpoint": new_human_poses[j],
                            "unseenFor": 0,
                            "firstSeenNearDoor": intersects_with_door(np_boxes[j]),
                            "seenFor": 0,
                            "box": np_boxes[j],
                            "wasCounted": False}
                
                self.humans.set('pred_classes', torch.as_tensor(np.array([int(new_id) for new_id in new_ids])))
                self.old_ids = new_ids
                self.data, actionFlag = update_people_count(self.tracked_humans, self.data, atLeastSeenFor, img_name)
                if view == "near" and actionFlag:
                    ViewImg(img_name, im, self.humans, self.old_humans, view, metadata)
            else:
                self.old_ids = []
                for i in range(len(self.humans)):
                    self.instance_count += 1
                    self.tracked_humans[str(self.instance_count)] = {
                            "midpoint": new_human_poses[i],
                            "unseenFor": 0,
                            "firstSeenNearDoor": intersects_with_door(np_boxes[i]),
                            "seenFor": 0,
                            "box": np_boxes[i],
                            "wasCounted": False}
                    self.old_ids.append(self.instance_count)
                    
                # Show the IoU with the door instead of the detection probability
                # self.humans.set('scores', torch.Tensor([IoU_door(self.tracked_humans[str(self.old_ids[i])]['box']) for i in range(len(self.humans))]))

                ids = torch.as_tensor(np.array([self.old_ids[i] for i in range(len(self.humans))]))
                self.humans.set('pred_classes', ids)
        
        if view == "all" or view == "humans" and len(self.tracked_humans) > 0:
            ViewImg(img_name, im, self.humans, self.old_humans, view, metadata)

            if slideshow:
                plt.show()
                clear_output(wait=True)
                sleep(0.1)

    def OnDetectionFinished(self, atLeastSeenFor, img_name, im, view, metadata):        
        self.data, actionFlag = final_update_people_count(self.tracked_humans, self.data, atLeastSeenFor, img_name)
        if view == "near" and actionFlag:
            ViewImg(img_name, im, self.humans, self.old_humans, view, metadata)
        return self.data

def SaveOutput(data, path, file_name, overwrite=False):
    if overwrite or 'cam5' + file_name + '.txt' not in listdir(path):
        print("Saving file ", (path + file_name + '.txt'))
        with open(path + file_name + '.txt', 'w') as outfile:
            json.dump(data, outfile)
