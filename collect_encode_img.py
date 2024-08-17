# import bchlib
import glob
import os
from PIL import Image,ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import pickle


def encode(args, secret):

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
    output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

    width = 400
    height = 400

    img_root_dir = args.images_dir
    files_list = [f for f in os.listdir(img_root_dir) if ".png" in f]

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        size = (width, height)
        for file in files_list:
            filename = os.path.join(img_root_dir, file)

            image = Image.open(filename).convert("RGB")
            image = np.array(ImageOps.fit(image,size),dtype=np.float32)
            image /= 255.

            feed_dict = {input_secret:[secret],
                         input_image:[image]}

            hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)

            rescaled = (hidden_img[0] * 255).astype(np.uint8)
            raw_img = (image * 255).astype(np.uint8)
            residual = residual[0]+.5

            residual = (residual * 255).astype(np.uint8)

            save_name = filename.split('/')[-1].split('.')[0]

            im = Image.fromarray(np.array(rescaled))
            im.save(args.save_dir + '/'+save_name+'_hidden.png')

            # im = Image.fromarray(np.squeeze(np.array(residual)))
            # im.save(args.save_dir + '/'+save_name+'_residual.png')
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='saved_models/stegastamp_pretrained')
    parser.add_argument('--dataset', type=str, default="COCO")
    args = parser.parse_args()
    
    # ====
    clean_data_root = os.path.join(
        "..", "DIP_Watermark_Evasion",
        "dataset", "Clean", args.dataset
    )
    save_dir = os.path.join(
        "..", "DIP_Watermark_Evasion",
        "dataset", "StegaStamp", args.dataset
    )
    save_encoder_dir = os.path.join(
        save_dir, "encoder_img"
    )
    os.makedirs(save_encoder_dir, exist_ok=True)

    args.images_dir = clean_data_root
    args.save_dir = save_encoder_dir

    secret = np.random.binomial(1, 0.5, 100)
    encode(args, secret)

    print(secret)
    save_secret_dir = os.path.join(save_dir, "encode_msg.pkl")
    with open(save_secret_dir, 'wb') as handle:
        pickle.dump(secret, handle, protocol=pickle.HIGHEST_PROTOCOL)