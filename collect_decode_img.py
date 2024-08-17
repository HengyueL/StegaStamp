from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import os
import pickle
import pandas as pd


def watermark_np_to_str(watermark_np):
    """
        Convert a watermark in np format into a str to display.
    """
    return "".join([str(i) for i in watermark_np.tolist()])


def decode(args):

    sess = tf.InteractiveSession(graph=tf.Graph())
    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    decode_img_dir = args.images_dir
    files_list = [f for f in os.listdir(decode_img_dir) if ".png" in f]

    res_dict = {
        'ImageName': [],
        "Encoder": [],
        'Decoder': []
    }

    msg_dir = args.msg_dir
    with open(msg_dir, 'rb') as handle:
        msg_np = pickle.load(handle)
    msg_str = watermark_np_to_str(msg_np)

    for file in files_list:
        filename = os.path.join(decode_img_dir, file)
        image = Image.open(filename).convert("RGB")
        image = np.array(ImageOps.fit(image,(400, 400)),dtype=np.float32)
        image /= 255.
        feed_dict = {input_image:[image]}
        secret = sess.run([output_secret],feed_dict=feed_dict)[0][0]
        packet_binary = "".join([str(int(bit)) for bit in secret])

        res_dict["ImageName"].append(file.replace("_hidden.png", ".png"))  
        res_dict["Decoder"].append([packet_binary])      
        res_dict["Encoder"].append([msg_str])

    save_csv_dir = msg_dir.replace("encode_msg.pkl", "water_mark.csv")
    df = pd.DataFrame(res_dict)
    df.to_csv(save_csv_dir, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='saved_models/stegastamp_pretrained')
    parser.add_argument('--dataset', type=str, default="COCO")
    args = parser.parse_args()

    image_root = os.path.join(
        "..", "DIP_Watermark_Evasion",
        "dataset", "StegaStamp", args.dataset
    )
    images_dir = os.path.join(
        image_root, "encoder_img"
    )
    encoding_dir = os.path.join(
        image_root, "encode_msg.pkl"
    )
    args.images_dir = images_dir
    args.msg_dir = encoding_dir
    decode(args)

