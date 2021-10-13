import json
import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from Detection import SaveOutput

def get_stops(data):
    predictedStops = []
    for k in data:
        if len(k) > 4 and k[-4:] == ".jpg":
            num_humans = data[k].get("Entered", 0) + data[k].get("Left", 0) + data[k].get("Unsure", 0)
            for i in range(num_humans):
                predictedStops.append(k)
#             if "Entered" in data[k] or "Left" in data[k] or "Unsure" in data[k]:
#                 predictedStops.append(k)
    return predictedStops

def get_stop_positions(stops):
    return [[int(k.split('_')[-2]),] for k in stops]

def ReadStops(file_name):
    f = open(file_name,'r')
    data = json.load(f)
    f.close()

    predictedStops = get_stops(data)
    return get_stop_positions(predictedStops), predictedStops, data

def read_classification(data, img):
    entered = data.get(img, {}).get("Entered", 0)
    left = data.get(img, {}).get("Left", 0)
    unsure = data.get(img, {}).get("Unsure", 0)
    return entered, left, unsure

def Classify(used_data_path, complete_dataset, overwrite=True):
    # save all detected bus stop images in a specified folder
    predictedStopLocations, predictedStops, pred_data = ReadStops(used_data_path + 'output.txt')
    stopLocations, stops, data = ReadStops(used_data_path[:-4] + 'BusStops.txt')
    
    # has a bus stop in general be correctly identified
    true_positives = []
    false_positives = []
    false_negatives = [] 
    
    # has a bus enter/leave been correctly identified
    true_positives_el = []
    false_positives_el = []
    
    if len(predictedStops) == 0:
        print("Info: No stops have been detected")
        false_negatives = stops
    elif len(stops) == 0:
        print("Info: No stops have been registered")
        false_positives = predictedStops
    elif len(predictedStopLocations) > 0 and len(stopLocations) > 0:
        C = cdist(predictedStopLocations, stopLocations)
    
        if len(stopLocations) - len(predictedStopLocations) > 0:
            print(len(C[0]))
            padding = np.zeros((len(stopLocations) - len(predictedStopLocations), len(C[0])))
            C = np.vstack((C, padding))
            print(len(C[0]))
        else:
            padding = np.zeros((len(C) , len(predictedStopLocations) - len(stopLocations)))
            C = np.hstack((C, padding))
    
        threshold = 4
        not_matches = np.where(C >= threshold)
        C[not_matches] = 10000000
    
        row_ind, assignment = linear_sum_assignment(C)
        cost = C[row_ind, assignment].sum()
    
        for i in range(len(assignment)):
            j = assignment[i]
            if i < len(predictedStopLocations) and j < len(stopLocations) and (predictedStopLocations[i][0] - stopLocations[j][0]) < threshold:
                print(predictedStops[i], " vs ", stops[j])
                true_positives.append(predictedStops[i])
    
                entered, left, unsure = read_classification(data, stops[j])
                entered_pred, left_pred, unsure_pred = read_classification(pred_data, predictedStops[i])
    
        #         entered_pred = data[predictedStops[i]].get("Entered", 0)
        #         entered = data[stops[j]].get("Entered", 0)
        #         left_pred = data[predictedStops[i]].get("Left", 0)
        #         left = data[stops[j]].get("Left", 0)
        #         unsure_pred = data[predictedStops[i]].get("Unsure", 0)
        #         unsure = data[stops[j]].get("Unsure", 0)
    
                if entered_pred == entered and left_pred == left and unsure_pred == unsure:
                    true_positives_el.append((predictedStops[i], entered_pred, entered, left_pred, left, unsure_pred, unsure))
                else:
                    false_positives_el.append((predictedStops[i], entered_pred, entered, left_pred, left, unsure_pred, unsure))
                    # or "Left" in data[k] or "Unsure" in data[k]:
            elif i < len(predictedStopLocations) and j < len(stopLocations):
                false_positives.append(predictedStops[i])
                false_negatives.append(stops[j])
    
                entered, left, unsure = read_classification(data, stops[j])
                entered_pred, left_pred, unsure_pred = read_classification(data, predictedStops[i])
        #         false_positives_el.append((predictedStops[i], entered_pred, 0, left_pred, 0, unsure_pred, 0))
        #         false_positives_el.append((stops[j], 0, entered, 0, left, 0, unsure))
            elif i < len(predictedStopLocations):
                false_positives.append(predictedStops[i])
    
        #         entered_pred, left_pred, unsure_pred = read_classification(data, predictedStops[i])
        #         false_positives_el.append((predictedStops[i], entered_pred, 0, left_pred, 0, unsure_pred, 0))
            else:
                false_negatives.append(stops[j])
                                         
    #         entered, left, unsure = read_classification(data, stops[j])
    #         false_positives_el.append((stops[j], 0, entered, 0, left, 0, unsure))
    len_true_negatives = len(complete_dataset) - len(true_positives) - len(false_positives) - len(false_negatives)
    
    print("True positives: ", len(true_positives))
    print("False positives: ", len(false_positives))
    print("False negatives: ", len(false_negatives))
    print("True negatives: ", len_true_negatives)
    
    print("True classifications: ", len(true_positives_el))
    print("False classifications: ", len(false_positives_el))
    
    results_dict = {
        "True positives": len(true_positives),
        "False positives": len(false_positives),
        "False negatives": len(false_negatives),
        "True negatives": len_true_negatives,
        "True classifications": len(true_positives_el),
        "False classifications": len(false_positives_el)
    }
    
    SaveOutput(results_dict, used_data_path, 'results', overwrite=True)
            
    return true_positives, false_positives, false_negatives, true_positives_el, false_positives_el, predictedStops
