from os import listdir, makedirs
from os.path import isfile, isdir, join
import json

from Detection import GetPredictor, Tracker, SaveOutput
from EvaluateResults import Classify, ReadStops
from MatchBusStops import MatchStops
from ExtractBagFile.ReadBagExtended import extract

import argparse

def get_folders(path):
    return sorted([join(path, f) for f in listdir(path) if isdir(join(path, f))])

def GetCompleteDataset(dataset_path):
    imgs = sorted([join(dataset_path, f) for f in listdir(dataset_path) if isfile(join(dataset_path, f)) and f[-4:] == ".jpg"])
    return [{"file_name" : f} for f in imgs]

"""
Allows to detect bus stops in all subfolders of the specified folder.
Overwrite determines if the detection should be repeated, if outputs of a previous run exist.
"""
def run_detections(used_data_path, overwrite=False): # TODO
    predictor = GetPredictor()
    folders = get_folders(used_data_path)
    tracker = Tracker()
    tracker.Init()
    for folder in folders:
        # print(listdir(folder))
        if overwrite or not "cam5output.txt" in listdir(folder):
            if "BusStops.txt" not in listdir(folder):
                print("Ground Truth not found, skipping " + folder)
                continue
            dataset_path = folder + "/cam5/"
            complete_dataset = GetCompleteDataset(dataset_path)
            json_data = tracker.DetectInDataset(complete_dataset, None, predictor, len(complete_dataset), start=0, step=1, slideshow=False, deleteAfter=1, view="none")

            print("Entered: ", json_data["Entered"])
            print("Left: ", json_data["Left"])
            print("Unsure: ", json_data["Unsure"])

            SaveOutput(json_data, folder, '/cam5output', overwrite=True)
        else:
            print("Output for ", folder, " already exists")

"""
Compares the detected output with the Ground Truth, if the ground truth exists.
The ground truth has to be in the same folder as the folder containing the images and needs to be called BusStops.
For the format, please check the existing files.
Overwrite determines if the evaluation should be repeated, if outputs of a previous run exist.
"""
def compute_results(used_data_path, overwrite):
    folders = get_folders(used_data_path)
    for folder in folders:
        files = listdir(folder)
        if overwrite or not "cam5results.txt" in files:
            if "cam5output.txt" in files and "BusStops.txt" in files:
                path = folder + "/cam5"
                dataset_path = folder + "/cam5/"
                complete_dataset = GetCompleteDataset(dataset_path)
                true_positives, false_positives, false_negatives, true_positives_el, false_positives_el, predictedStops = Classify(path, complete_dataset)
            else:
                print("No ground truth or output in Folder, skipping ", folder)
        else:
            print("Results for ", folder, " already exist, skipping")

"""
Compares the detected stop locations in the folder used_data_path with the official data where the bus stops are located.
Detections further away then 100m are categorized as unofficial, below this distance as official.
The images are saved in corresponding folders at the specified output_path.
"""
def categorize_images(used_data_path, output_path):
    # output_dir = output_path + "DetectedBusStops/" + used_data_path.split('/')[-2]
    # output_dir_officials = output_path + "DetectedOfficialBusStops/" + used_data_path.split('/')[-2]
    # output_dir_unofficials = "DetectedUnofficialBusStops/" + used_data_path.split('/')[-2]
    folders = get_folders(used_data_path)
    for folder in folders:
        files = listdir(folder)
        if "cam5output.txt" in files:
            predictedStopLocations, predictedStops, pred_data = ReadStops(folder + '/cam5output.txt')

            output_dir = output_path + "DetectedBusStops/"
            output_dir_officials = output_path + "DetectedOfficialBusStops/"
            output_dir_unofficials = output_path + "DetectedUnofficialBusStops/"
            MatchStops(predictedStops, output_dir, output_dir_officials, output_dir_unofficials, True)

            output_dir += folder.split('/')[-1]
            output_dir_officials += folder.split('/')[-1]
            output_dir_unofficials += folder.split('/')[-1]
            MatchStops(predictedStops, output_dir, output_dir_officials, output_dir_unofficials)
        else:
            print("Output in Folder, skipping ", folder)

"""
Accumulates the data that can be found in subfolders of the given path
and saves the result in the output folder as a file called 'results.txt'.
"""
def accumulate_results(used_data_path, output_folder):
    acc_data = {}
    folders = get_folders(used_data_path)
    for folder in folders:
        if "cam5results.txt" in listdir(folder):
            with open(folder + "/cam5results.txt",'r') as results_file:
                data = json.load(results_file)
            for key in data:
                acc_data[key] = acc_data.get(key, 0) + data[key]
            acc_data["CountedDatasets"] = acc_data.get("CountedDatasets", 0) + 1
    
    makedirs(output_folder, exist_ok=True)
    with open(output_folder + 'results.txt', 'w') as outfile:
        json.dump(acc_data, outfile)
    print("Finished Accumulating results!")

"""
 This is a function that helps identifying the reasons why unofficial bus stops occurred.
 It allows to extract a bag file for a given image folder found at result_path, if the bag file can be found under the bag_path.
 A number of images before and after a person entered the bus then get saved in the output_dir, for the specified cameras.
 Taking a look at the output images can then help to identify what caused this unofficial bus stop.
"""
def extract_unofficial_busstops(output_dir, result_path, bag_path, cams):
    folders = get_folders(result_path)
    bag_folders = listdir(bag_path)
    print(bag_folders)
    for folder in folders:
        formatted_folder = (folder.split('/')[-1]).replace('-', '_')
        # the following line might need to be used for some data, if the output is stored under cam5outputs
        # formatted_folder = (folder.split('/')[-1])[4:].replace('-', '_')
        folder_path = folder
        imgs = sorted([f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f[-4:] == ".jpg"])
        img_times = [int(f.split('_')[-2]) for f in imgs]
        if formatted_folder in bag_folders:
            print("Extracting dataset", formatted_folder)
            extract([], output_dir + folder, bag_path + formatted_folder, img_times, 30, False, cams)
        else:
            print("Could not detect bags for", formatted_folder)
            print("imgs:", str(img_times))

def main():
    parser = argparse.ArgumentParser(description="Extract images and GPS from a rosbag.")
    parser.add_argument(
        "-i", "--input-folder", default='../data/extracted_images/', help="The folder in which all camera images are stored in the subdirectories")
    parser.add_argument(
        "-b", "--bag-folder", default='../data/', help="The folder in which all ros bag files with camera footage are stored in the subdirectories")
    parser.add_argument(
        "-o", "--output_folder", default='../results/', help="The directory in which the outputs get stored.")
    parser.add_argument(
        "-O", "--overwrite", action='store_true',
        help="Decides if processing steps should get performed if the output already exists.")
    parser.add_argument(
        "-d", "--detect_stops", action='store_true',
        help="Runs a detection on all available datasets in the input path.")
    parser.add_argument(
        "-e", "--evaluate_stops", action='store_true',
        help="Evaluates all computed output with manual labels data, if manual labels are available.")
    parser.add_argument(
        "-a", "--accumulate_results", action='store_true', help="Accumulates all available results for the datasets.")
    parser.add_argument(
        "-c", "--categorize_images", action='store_true', help="Categorizes all detected bus stops in the output into whether they are near an existing busstop or not.")
    parser.add_argument(
        "-x", "--extract_unofficial_busstops", action='store_true', help="Extracts images for all unofficial bus stops if a stop is in one of the bags in the bag file directory.")
    parser.add_argument(
        "-C", "--cam-id", nargs='+', type=int, default=[3,], help="Selected camera IDs to extract")

    args = parser.parse_args()

    if args.detect_stops:
        print("Detecting bus stops... this might take a while")
        run_detections(args.input_folder, args.overwrite)

    if args.evaluate_stops:
        print("Evaluating the computed output with Ground Truth data...")
        compute_results(args.input_folder, args.overwrite)

    if args.accumulate_results:
        print("Accumulating results...")
        accumulate_results(args.input_folder, args.output_folder)

    if args.accumulate_results:
        print("Categorizing all detected bus stops in the output into whether they are near an existing busstop or not...")
        categorize_images(args.input_folder, args.output_folder)

    if args.extract_unofficial_busstops:
        print("Storing images for all detected unofficial bus stops... this might take a while")
        extract_unofficial_busstops(args.output_folder + "UnofficialBusstops/", args.output_folder + "DetectedUnofficialBusStops/", args.bag_folder, [3, 5])

if __name__ == '__main__':
    main()

