import sys
from os import listdir
from os.path import isfile, join
import json
import cv2
import matplotlib.pyplot as plt

# print('Argument List:', str(sys.argv))
used_data_path = sys.argv[1]
print("used_data_path", used_data_path)

imgs = sorted([join(used_data_path, f) for f in listdir(used_data_path) if isfile(join(used_data_path, f)) and f[-4:] == ".jpg"])
# print(imgs[:10])

json_data = {}
fig = plt.figure(figsize=(128, 72), dpi=10)
axe = plt.gca()

images = []

i = 0
while 0 <= i < len(imgs):
   print(i)
   img_name = imgs[i] 

   
   axe.set_title(img_name)
   img = cv2.imread(img_name)
   plt.imshow(img) # [:, :, ::-1])
   plt.show(block=False)
   plt.pause(0.1)

   inp = input()
   if inp == "s":
      break
   elif inp == "e":
      json_data[img_name] = {"Entered": 1},
      i += 1
   elif inp == "l":
      json_data[img_name] = {"Left": 1},
      i += 1
   elif inp == "u":
      json_data[img_name] = {"Unsure": 1},
      i += 1
   elif inp == "" or inp == '\x1b[C':
      i += 1
   elif inp == "b" or inp == '\x1b[D':
      i -= 1
   else:
      print("Invalid input " + inp)

with open(used_data_path[:-5] + 'BusStops.txt', 'w') as outfile:
    json.dump(json_data, outfile)
print("Saved to ", used_data_path[:-5] + 'BusStops.txt')
