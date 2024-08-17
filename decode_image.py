from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import argparse
import os
import pickle


def decode(args):

    sess = tf.InteractiveSession(graph=tf.Graph())
    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    filename = args.decode_img_dir
    image = Image.open(filename).convert("RGB")
    image = np.array(ImageOps.fit(image,(400, 400)),dtype=np.float32)
    image /= 255.

    feed_dict = {input_image:[image]}

    secret = sess.run([output_secret],feed_dict=feed_dict)[0][0]

    packet_binary = "".join([str(int(bit)) for bit in secret])
    print("save {}".format(packet_binary))
    with open(args.msg_save_dir, 'wb') as handle:
        pickle.dump(packet_binary, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--decode_img_dir', type=str, default=None)
    parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument("--msg_save_dir", type=str, default="")
    args = parser.parse_args()
    decode(args)

