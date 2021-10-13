# BusStopDetection
This Project allows to detect locations in which people enter or exit a bus, based on camera footage of the bus. The detected locations can then be compared to the location of the officially registered stops by the bus stop company. The detected stops then get devided into "official" and "unofficial" stops, based on if the detected stop location is closer than 100m to the nearest listed bus stop. An image for each detection is stored and can be uploaded e.g. to AgileMapper https://app.agilemapper.com/maps for visualization. 

![Alt text](images/Detection.png?raw=true "Humans detected in Bus Camera Footage")

## Install Instructions
- Clone this repository
- Install Detectron2 (https://detectron2.readthedocs.io/en/latest/tutorials/install.html) and all requirements of it. This project was tested using Python 3.7, Pytorch 1.9 and Cuda version 1.1, so we recommend using the same or similar versions

## Usage Instructions
The project currently assumes that data from the bus is stored in ros bag files, as produced by the BusEdge system https://github.com/CanboYe/BusEdge.

### Extracting Images from a bag File
The script scripts/ExtractBagFile/read_bag_stops allows to extract all images of a bag file or all ros bag files in a folder into a new folder, while filtering out images with a high velocity or at the bus depot location. For the detection, camera images from the right-back camera are needed (id 5). So in order to extract these images, open a terminal in the scripts/ExtractBagFile folder and run:
`python3 read_bag_stops.py -f path_to_folder_with_bags -o path_output -g -c 5`.

### Running the Detector on the Images
The script scripts/BusStopUtilities allows to move the data through a pipeline of detecting bus stops, categorizing and visualizing them. By calling the script with different flags, different steps can be executed. Some flags are also needed to specify some paths that are needed for some of the processing steps. The paths that can be specified are:
- -i / --input-folder:  The folder in which all camera images are stored in the subdirectories
- -b / --bag-folder:    The folder in which all ros bag files with camera footage are stored in the subdirectories
- -o / --output_folder: The directory in which the outputs get stored.

Additionally, there is the override flag:
- -O / --overwrite:     Decides if processing steps should get performed if the output already exists for detection and evaluation.

Following processing steps can than be executed:
- -d / --detect_stops:                Runs a detection on all available datasets in the input path and stores the output in the input folders. 
- -e / --evaluate_stops:              Evaluates all computed output with manual labels, if manual labels are available. 
- -a / --accumulate_results:          Accumulates all available results for the datasets and stores it in the output folder.
- -c / --categorize_images:           Categorizes all detected bus stops in the output into whether they are near an existing busstop or not.
- -x / --extract_unofficial_busstops: Extracts images for all unofficial bus stops if a stop is in one of the bags in the bag file directory.
- -C / --cam-id:                      Selected camera IDs to extract from the bags if the -x flag is set

To run the detection on all available data with the standard paths, overwriting previous output and evaluating the results with manual labels, go the scripts folder and run the following command in a terminal:

`python3 BusStopUtilities.py -O -d -e -a -c -x -C 5`

The results can be found inthe folder results, if no other path has been specified. In the sample data, only one case for an unofficial bus stop exists and no bag file for the data is provided. If no changes have been performed for the repository, the output should have the following structure:
\results
  \DetectedBusStops
    \SampleData
      camera5_1627665686_867068122.jpg
    camera5_1627665686_867068122.jpg
  \DetectedUnofficialBusStops
    \SampleData
      camera5_1627665686_867068122.jpg
    camera5_1627665686_867068122.jpg
  \results.txt
The output in the terminal should look similar to data/extracted_images/SampleData/SampleOutput.txt.

Not all flags need to be set, so for example to run only the detection, run the following command instead:

`python3 BusStopUtilities.py -O -d`

### Alternative 1:
Alternatively to the approach on top, these steps can be called in the notebook HumanDetector.ipynb. While the previous method helped to process many datasets at once, this method allows to debug the code and run it on single datasets or even parts of datasets.

### Alternative 2 (incomplete):
Another approach is to have a subscriber listen to ros topics of the data. The data can come from a ros bag that gets read and published, e.g. by running `rosbag play <your bagfile>` in another terminal. The data can also come from a live source, e.g. the sensors itself. This means that in theory this approach can be used to detect bus stops live while the bus is driving. Currently, only the detection can be performed, but the data is not further processed (e.g. no classification is performed). Furthermore, as the velocity data from the bus is in a special format that deviates from the nmea_navsat_driver. This means, that the project has to run in a working ros workspace in which the modified message has to be build and sourced. While in principle possible, we did not try this out and ignore the velocity in the current version of the subscriber. To run the current version of the subscriber, go to the scripts directory and run

`python3 BusstopDetectionSubscriber.py`

### Visualizing the Detection Locations
The location of the detected stops can be visualized using AgileMapper https://app.agilemapper.com/maps. For this, create a new, free account and create a new map. Take all detections you want to have on the map, e.g. all images in the results/UnofficialBusStops folder and upload them to the map. When the upload is finished, they will be shown on the map, according to their GPS location.

![Alt text](images/MapStops.png?raw=true "Detections visualized on AgileMapper")

## How it works
### Data filters
We noticed during development that other sensors than the camera can help filtering the data efficiently, so that the main detection by camera only has to run on a subset of all data of the bus. We used the GPS data to filter out locations in which the bus is not on duty, e.g. because the bus driver is on break or hasn't left the depot yet. We ommit images from 2 known break locations and from a gas station where the bus usually refills the gas. Furthermore, we use the velocity data from the IMU of the bus to check if it is currently driving at a high velocity. We assume that no human will enter or exit the bus at a high velocity. 

### Human Detection
We use the standard object detection by Detectron2, a trained R-CNN, to detect humans. The predictor can detect 81 different classes, but as this project only needs to detect the positions of humans, we discard most of the data. We noticed that wrong detections occur in the mirror of the bus and in the windows. To counteract wrong detections in the mirror, we exclude detections if they have a high overlap with the typical position of the mirror in the images. The area of the mirror and the threshold for the overlap are defined in the script scripts/Detector.py in the method 'intersects_with_mirror' and can be visualized in the method 'ViewImg' in the same script. We noticed that the area in which the mirror is can change between runs, so modifications might be needed for some data. To counteract wrong detections in the window, we blacked out the left pixel columns of the image, in which the window is located. 

### Tracking the Humans
Our tracking based on the Hungarian algorithm. By utilizing the IoU (Intersection over Union), we check if a human intersects with the positions in the last frame, in case multiple humans overlap between the frames, we take the largest intersection in total. For the intersection, we use the boundingbox of the humans. By storing the locations of humans from previous frames and also checking those with the current frame, the tracking algorithm is able to reidentify a human with the same id as before, even if the human was not detected in a few frames.

### Detecting if People are entering or exiting
To detect if people are entering or exiting the bus, we are checking if a person is overlapping with the area in which the door is located. The area and the threshold for the overlap for the door are defined in the script scripts/Detector.py in the method 'intersects_with_door' and can be visualized in the method 'ViewImg'. Changing this area and threshold has a great influence on the detection. We tuned these parameters on a number of datasets. However, by using more data and better tuning of these parameters, better results might be able to achieve, e.g. to differentiate if someone is walking close before the door or in the door. The algorithm stores if a person is detected in the first two frames near the door. If the person then leaves this area, he is considered as a leaving passenger, for the frame in which he leaves the area. If the person is last detected near the door, we classify the person as "unsure", as he might have either entered or exited the bus and the algorithm has problems deciding. This typically occurs when someone comes to the bus in front of it or leaves in front of the bus. If a person is first detected outside the door area, but then moves into the door area, we classify this as entering. 

### Classifying a stop into unofficial or official
In the text file data/stops.txt, all bus stops and their GPS location for the bus route are stored, as reported by the bus company. We read in this data and check for each detected stop, which official stop is closest to the current stop. If the closest stop is closer than 100m, we classify the stop as official, else as unofficial. 

## Evaluation
To test how well the algorithm performs, we annotated 9 datasets by hand with labels in which frame somebody enters or exists the bus, and compared them to the detections of our algorithm on the same data. This evaluation is by no means scientific and the results might be inaccurate, so it should be only seen as a general idea on how well the algorithm performs.

We ran the detection and compared the times in which the detection detects somebody entering or exiting, to the labeles we created by hand. We find the best matching for both times, and check for each stop if the match is closer together than 4 seconds. This approach seems to work well in most cases. The algorithm then checks for each positive matching (detection and manual label are closer than 4s) if the classification "enter", "leave" or "unsure" is the same. The classification "unsure" is currently only used by the detector and will always yield in a false classification. However, for the end user, such a classification might be more useful than a wrong one. The algorithm then aggregates the results for each detected and undetected stop and all classifications and stores the result in a text file.

Notes:
- Our tests have been performed mainly on the same data that we used to build our detection. However, we did not see mayor differences between unseen and seen data during evaluation.
- The evaluation only checks if a person enteres or exists the bus. It makes no statement whether this actually means that it is a bus stop. Our hand labels also consider all people entering or exiting the bus, not only those who actually ride the bus. In example, this means that whenever the bus driver enters or exists the bus, this is counted as a bus stop. 
- In the test data only images in which the bus is moving slow or is standing have been considered. Furthermore, images from 2 major pause locations of the bus driver have been excluded. However, stops at the gas station have been considered. The newest version will exclude them, but the evaluation hasn't been run again after excluding the gas stations. 
- Our evaluation algorithm assumes that there is always at most one person per frame entering or exiting the bus. This is not always the case for the detection, so the evaluation might classify these situations wrong.
- The data contains mainly data from daytime, but also nighttime. The human detection works a lot better during daytime, but the algorithm is still able to detect some bus stops at night. 

![Alt text](images/ResultsDetection.png?raw=true "Detection Results")

![Alt text](images/ResultsClassification.png?raw=true "Classification Results")

We checked the wrong detections and classifications to learn about shortcomings of our algorithm. Following is a summary of our observations:

False Negatives in Detection:
- 5x humans were not detected due to bad image quality in darkness (i.e. b/w images). The humans are even hard to detect for a human in some frames.
- a man exits the bus but then reenters it, which we labelled as two detections, but the detector labeled it as one
- 2x Bus Driver at Gas station is missdetected
- A sign is between the camera and the entering people, so only 4 of 6 people got detected in this case

False Positives in Detection:
- Bus Driver at Gas station is missclassified
- Man talked to bus driver but didn’t enter, which we didn't label as a stop, but the algorithm did

False Classifications:
- Person comes from in front of the bus (often classified as unsure)
- Person leaves in front of the bus
- Wrong detection of the position of the exiting person when only parts of the person are visible yet
- A sign is between the camera and the entering people
- Evaluation algorithm has problems judging when 2 enters/leaves are detected in the same picture

## The Project
I developed this project during a 2 month internship at the NavLab at Carnegie Mellon University in Pittsburgh. A huge thanks goes out to my advisor, Dr. Cristoph Mertz, for his great support for this internship.

