"""
    This script is use dwtDctSvd/rivaGan to decode clean images to compute FPR.
"""
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import os, argparse
import numpy as np
import pandas as pd
from general import  watermark_str_to_numpy
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from PIL import Image, ImageOps


def main(args):
    # === Set the dataset paths ===
    dataset_input_path = os.path.join(
        args.clean_data_root, args.dataset
    )
    # === Scan all images in the dataset (clean) ===
    img_files = [f for f in os.listdir(dataset_input_path) if ".png" in f]
    print("Total number of images: [{}] --- (Should be 100)".format(len(img_files)))

    output_root_path = os.path.join(
        "..", "DIP_Watermark_Evasion", "dataset", 
        "Clean_Watermark_Evasion", args.watermarker, args.dataset
    )
    os.makedirs(output_root_path, exist_ok=True)

    # === GT waternark ===
    print("Watermarker: ", args.watermarker)
    watermarked_file = os.path.join(
        "..", "DIP_Watermark_Evasion", 
        "dataset", args.watermarker, args.dataset, "water_mark.csv"
    )
    watermarked_data = pd.read_csv(watermarked_file)
    watermark_gt_str = watermarked_data.iloc[0]["Encoder"]
    if watermark_gt_str[0] == "[":  # Some historical none distructive bug :( will cause this reformatting
        watermark_gt_str = eval(watermark_gt_str)[0]
    watermark_gt = watermark_str_to_numpy(watermark_gt_str)

    # === Init watermarker ===
    sess = tf.InteractiveSession(graph=tf.Graph())
    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)
    
    # Init the dict to save watermarking summary
    save_csv_dir = os.path.join(output_root_path, "water_mark.csv")
    res_dict = {
        "ImageName": [],
        "Decoder": [],
    }

    for img_name in img_files:
        img_clean_path = os.path.join(dataset_input_path, img_name)
        print("***** ***** ***** *****")
        print("Processing Image: {} ...".format(img_clean_path))

        image = Image.open(img_clean_path).convert("RGB")
        image = np.array(ImageOps.fit(image,(400, 400)),dtype=np.float32)
        image /= 255.
        feed_dict = {input_image:[image]}
        secret = sess.run([output_secret],feed_dict=feed_dict)[0][0]
        packet_binary = "".join([str(int(bit)) for bit in secret])

        watermark_decode_str = packet_binary

        res_dict["ImageName"].append(img_name)
        res_dict["Decoder"].append([watermark_decode_str])

    df = pd.DataFrame(res_dict)
    df.to_csv(save_csv_dir, index=False)


if __name__ == "__main__":
    print("Use this script to download DiffusionDB dataset.")
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        '--clean_data_root', type=str, help="Root dir where the clean image dataset is located.",
        default=os.path.join("..", "DIP_Watermark_Evasion", "dataset", "Clean")
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str, help="The dataset name: [COCO, DiffusionDB]",
        default="COCO"
    )
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, help="Specification of watermarking method. ['dwtDctSvd', 'rivaGan']",
        default="StegaStamp"
    )
    parser.add_argument('--model', type=str, default='saved_models/stegastamp_pretrained')
    args = parser.parse_args()
    main(args)
    print("Completd")