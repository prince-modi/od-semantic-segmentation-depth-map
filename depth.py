"""Compute depth maps for images in the input folder.
"""
import sys
sys.path.append('MiDaS')
import os
import glob
import torch
import modified_utils as utils
import cv2
import argparse
import time
import shutil

import numpy as np

from imutils.video import VideoStream
from midas.model_loader import default_models, load_model


first_execution = True
def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


def create_side_by_side(image, depth, grayscale):
    """
    Take an RGB image and depth map and place them side by side. This includes a proper normalization of the depth map
    for better visibility.

    Args:
        image: the RGB image
        depth: the depth map
        grayscale: use a grayscale colormap?

    Returns:
        the image and depth map place side by side
    """
    print
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3

    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    if image is None:
        return right_side
    else:
        return np.concatenate((image, right_side), axis=1)

class Depth:
    def __init__(self) -> None:
        print("Initialising model")
        
        # set torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.model_type="dpt_beit_large_512"
        # self.model_type="dpt_swin2_tiny_256"
        # self.model_type="dpt_swin2_large_384"
        weights_path = default_models[self.model_type]
        model_path = "MiDaS/" + weights_path
        self.optimize, self.side, height, square=False, False, None, False
        self.grayscale=True
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: %s" % self.device)
        self.model, self.transform, self.net_w, self.net_h = load_model(self.device, model_path, self.model_type, self.optimize, height, square)


    def inference_depth(self,frame_path=None,save=True, show=True, output_path="output", send_depth=False):
        t1 = time.time()

        #preprocessing
        if frame_path == None:
            frame_path = "frames/0.jpg"
            print(f"no frame path mentioned. Inferencing on the image: {frame_path}")
        original_image_rgb = utils.read_image(frame_path) # in [0, 1]
        image = self.transform({"image": original_image_rgb})["image"]
    
        # compute
        t2 = time.time()
        with torch.no_grad():
            prediction = process(self.device, self.model, self.model_type, image, (self.net_w, self.net_h), original_image_rgb.shape[1::-1],
                                    self.optimize, False)

        # output
        t3 = time.time()
        filename = os.path.join(output_path, "depth", os.path.splitext(os.path.basename(frame_path))[0] + '-' + self.model_type)
                
        if send_depth:
            out, depth = utils.write_depth(filename, prediction, self.grayscale, bits=2, return_depth=send_depth)
        else:
            out = utils.write_depth(filename, prediction, self.grayscale, bits=2, return_depth=send_depth)

        out = cv2.resize(out, (960, 720), interpolation = cv2.INTER_AREA)
        
        if save:
            cv2.imwrite(filename + ".png", out)
        if show:
            cv2.imshow('depth_image', out)
            cv2.waitKey(1)
             
        t4 = time.time()
        # print(f'=========inf time:{t3-t2}\n')
        inf_time = t3-t2
        if send_depth:
            return out, depth
        return out


if __name__=="__main__":
    inf_row = []
    obj = Depth()
    i = 0

    input = "ABS_calculation_dataset"
    if input is not None:
        image_names = sorted(glob.glob(os.path.join(input, "*")))
        num_images = len(image_names)

    # while True:
    for i, temp in enumerate(image_names, start=0):
        # frame_path = f"ABS_calculation_dataset/{i}.jpg"
        try:
            print(temp)
            output = obj.inference_depth(temp, show=True, save=True, output_path="results_ABS_2") 
            # inf_row.append(inf_time)
            i += 1
        except Exception as e:
            print(e)
            break
    # import csv
    # with open('depth_results.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["SNo", "Time"])
    #     for i ,row in enumerate(inf_row):
    #         writer.writerow([i,row])