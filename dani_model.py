import torch
model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')  # custom trained model
import os
import cv2

# # Images


# # Inference
# results = model(im)

# # Results
# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
# results.save()  # or .show(), .save(), .crop(), .pandas(), etc.

#im = "D:\Testing-Miniapi\Back-End-Flask\images/0001.jpg"

# results.xyxy[0]  # im predictions (tensor)
# results.pandas().xyxy[0]  # im predictions (pandas)

def get_var_value(filename="varstore.dat"):
    with open(filename, "a+") as f:
        f.seek(4)
        val = int(f.read() or 4) + 1
        f.seek(4)
        f.truncate()
        f.write(str(val))
        val = str(val)
        path = os.getcwd() + '/runs\detect\exp' +  val
        return path




def image_TBC_location(im, name):   
    results = model(im)
    # Results
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    results.save()  # or .show(), .save(), .crop(), .pandas(), etc.


    results.xyxy[0]  # im predictions (tensor)
    results.pandas().xyxy[0]  # im predictions (pandas)
    your_counter = get_var_value()
    final_path =  your_counter + '/' + name
    print(final_path)
    return final_path

#image_TBC_location(im, ".")


# final_path =  your_counter + '/tb0005_png_jpg.rf.7523f94c20ea8e02fc836b3b7e576648.jpg'
# # print("This script has been run {} times.".format(your_counter))
# print(final_path)

# This script has been run 1 times
# This script has been run 2 times
# etc.

# i = 11
# i+=1

# path = os.getcwd() + '/runs\detect\exp' + str(i) + '/tb0005_png_jpg.rf.7523f94c20ea8e02fc836b3b7e576648.jpg'
# print(path)
# New_Img_name = str(Instance_Number - 1).zfill(4) + '.jpg'
# path = path + '/' + New_Img_name
# print(path)
# cv.imwrite(path, New_Img)
# return path
