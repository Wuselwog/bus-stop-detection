import cv2
from detectron2.utils.visualizer import Visualizer, ColorMode
import matplotlib.pyplot as plt

from Detection import Tracker

def VisualizeData(dataset_dicts, metadata, predict=None, num_imgs=10, start=0, step=4, sample_random=False, slideshow=False):
    img_fig = None
    
    if sample_random:
        selected_data = random.sample(dataset_dicts, num_imgs)
    else:
        selected_data = dataset_dicts[start:start + step * num_imgs:step]
    for d in selected_data:
        im = cv2.imread(d["file_name"])
        
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata, 
                       scale=0.5, 
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        
        if predict is None:
            out = v.draw_dataset_dict(d)
        else:
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        plt.figure(figsize=(128, 72), dpi=10)
        plt.title(d["file_name"])
        img_fig = plt.imshow(out.get_image()[:, :, ::-1])
        
        if slideshow:
            plt.show()
            clear_output(wait=True)
            sleep(0.5)

def VisualizeCase(complete_dataset, predictor, case_id, cases, start, length):
    if 0 > case_id > len(cases):
        print("Invalid ID")
        return
    img_id = complete_dataset.index({'file_name': cases[case_id]}) # [0]
    tracker = Tracker()
    tracker.Init()
    json_data = tracker.DetectInDataset(complete_dataset, None, predictor, length, start=img_id - start, step=1, slideshow=False, deleteAfter=1, view="humans")
    # json_data = TrackData(complete_dataset, None, predictor, length, start=img_id - start, step=1, slideshow=False, deleteAfter=1, view="humans")
    print("Entered: ", json_data["Entered"])
    print("Left: ", json_data["Left"])
    print("Unsure: ", json_data["Unsure"])

def ShowImages(complete_dataset, data, imgs, predictor=None):
    for img_name in imgs:
        if type(img_name) == tuple:
            print(img_name)
            img_name, entered_pred, entered, left_pred, left, unsure_pred, unsure = img_name
            if entered_pred:
                print("Pred Enter")
            if left_pred:
                print("Pred Left")
            if unsure_pred:
                print("Pred Unsure")
            if entered:
                print("Truth Enter")
            if left:
                print("Truth Left")
            if unsure:
                print("Truth Unsure")
        idx = complete_dataset.index({'file_name': img_name})
        print(idx, " - ", img_name)
        humans = {}
        img = cv2.imread(img_name)
        
#         # increase the contrast of the images
#         img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#         # equalize the histogram of the Y channel
#         img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
#         # convert the YUV image back to RGB format
#         img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
#         img = cv2.equalizeHist(img)
        VisualizeData(complete_dataset, None, predict=predictor, num_imgs=1, start=idx, step=1, sample_random=False, slideshow=False)
        # ViewImg(img_name, img, detectron2.structures.instances.Instances([]), {}, "all", None)