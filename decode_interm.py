"""
    This script is a skeleton file for **Taihui** to:

    1) Read in the watermark evasion interm. results

    2) Decode each of the interm. result using the encoder/decoder API

    3) Save the result with standardized format
"""
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)

import argparse, os, cv2
import pickle5 as pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
# =====
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from general import watermark_str_to_numpy, watermark_np_to_str, uint8_to_float, compute_ssim
from PIL import Image, ImageOps


def calc_mse(img_1_bgr_uint8, img_2_bgr_uint8):
    img_1_float = uint8_to_float(img_1_bgr_uint8)
    img_2_float = uint8_to_float(img_2_bgr_uint8)
    mse = np.mean((img_1_float - img_2_float)**2)
    return mse


def main(args):
    # === This is where the interm. results are saved ===
    data_root_dir = os.path.join(
        "..", "DIP_Watermark_Evasion",
        "Result-Interm", args.watermarker, 
        args.dataset, args.evade_method, args.arch
    )
    file_names = [f for f in os.listdir(data_root_dir) if ".pkl" in f]  # Data are saved as dictionary in pkl format.

    # === This is where the watermarked image is stored ===
    im_w_root_dir = os.path.join(
        "..", "DIP_Watermark_Evasion",
        "dataset", args.watermarker, args.dataset, "encoder_img"
    )
    # === This is where the original clean image is stored ===
    im_orig_root_dir = os.path.join(
        "..", "DIP_Watermark_Evasion",
        "dataset", "Clean", args.dataset
    )

    # === Save the result in a different location in case something went wrong ===
    save_root_dir = os.path.join(
        "..", "DIP_Watermark_Evasion",
        "Result-Decoded", args.watermarker, 
        args.dataset, args.evade_method, args.arch
    )
    os.makedirs(save_root_dir, exist_ok=True)
    
    # === Process each file ===
    for n_file, file_name in enumerate(file_names):
        if n_file < args.start:
            print("Skip {}".format(file_name))
        elif n_file >= args.end:
            return
        else:
            # Retrieve the im_w name
            im_w_file_name = file_name.replace(".pkl", ".png")
            if "_hidden" in im_w_file_name:
                im_orig_name = im_w_file_name.replace("_hidden", "")
            else:
                im_orig_name = im_w_file_name

            # Readin the intermediate files
            data_file_path = os.path.join(data_root_dir, file_name)
            print(data_file_path)
            with open(data_file_path, 'rb') as handle:
                data_dict = pickle.load(handle)
            # Readin the im_w into bgr uint8 format
            im_w_path = os.path.join(im_w_root_dir, im_w_file_name)
            im_w_bgr_uint8 = cv2.imread(im_w_path)
            # Readin the 
            im_orig_path = os.path.join(im_orig_root_dir, im_orig_name)
            im_orig_bgr_uint8 = cv2.imread(im_orig_path)
            if args.watermarker == "StegaStamp" and args.arch in ["cheng2020-anchor", "mbt2018"]:
                im_orig_bgr_uint8 = cv2.resize(im_orig_bgr_uint8, (400, 400), interpolation=cv2.INTER_LINEAR)
            
            # Get the reconstructed data from the interm. result
            if args.evade_method == "WevadeBQ":
                img_recon_list = data_dict["best_recon"]
            else:
                img_recon_list = data_dict["interm_recon"]  # A list of recon. image in "bgr uint8 np" format (cv2 standard format)
            n_recon = len(img_recon_list)
            print("Total number of interm. recon. to process: [{}]".format(n_recon))

            # === Initiate a encoder & decoder ===
            sess = tf.InteractiveSession(graph=tf.Graph())
            model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)
            input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
            input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)
            output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
            output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

            watermark_gt_str = data_dict["watermark_gt_str"]
            if watermark_gt_str[0] == "[":  # Some historical none distructive bug :( will cause this reformatting
                watermark_gt_str = eval(data_dict["watermark_gt_str"])[0]
            watermark_gt = watermark_str_to_numpy(watermark_gt_str)

            # Process each inter. recon
            watermark_decoded_log = []  # A list to save decoded watermark
            index_log = data_dict["index"]
            psnr_orig_log = []
            mse_orig_log = []
            psnr_w_log = []
            mse_w_log = []
            ssim_orig_log = []
            ssim_w_log = []
            for img_idx in range(n_recon):
                img_bgr_uint8 = img_recon_list[img_idx]    # shape [512, 512, 3]
                if args.watermarker == "StegaStamp" and args.arch in ["cheng2020-anchor", "mbt2018"]:
                    img_bgr_uint8 = cv2.resize(img_bgr_uint8, (400, 400), interpolation=cv2.INTER_LINEAR)

                # =================== YOUR CODE HERE =========================== #
                
                # Step 0: if you need to change the input format
                img_input = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_input)
                image = np.array(ImageOps.fit(image,(400, 400)),dtype=np.float32)
                image /= 255.

                # Step 1: Decode the interm. result
                feed_dict = {input_image:[image]}
                secret = sess.run([output_secret],feed_dict=feed_dict)[0][0]
                watermark_decoded_str = "".join([str(int(bit)) for bit in secret])

                # Step 2: log the result
                watermark_decoded_log.append(watermark_decoded_str)

                # ============================================================= #

                # Calculate the quality: mse and psnr
                mse_recon_orig = calc_mse(im_orig_bgr_uint8, img_bgr_uint8)
                mse_recon_w = calc_mse(im_w_bgr_uint8, img_bgr_uint8)

                psnr_recon_orig = compute_psnr(
                    im_orig_bgr_uint8.astype(np.int16), img_bgr_uint8.astype(np.int16), data_range=255
                )
                psnr_recon_w = compute_psnr(
                    im_w_bgr_uint8.astype(np.int16), img_bgr_uint8.astype(np.int16), data_range=255
                )
                ssim_recon_orig = compute_ssim(
                    im_orig_bgr_uint8.astype(np.int16), img_bgr_uint8.astype(np.int16), data_range=255
                )
                ssim_recon_w = compute_ssim(
                    im_w_bgr_uint8.astype(np.int16), img_bgr_uint8.astype(np.int16), data_range=255
                )

                
                mse_orig_log.append(mse_recon_orig)
                mse_w_log.append(mse_recon_w)
                psnr_orig_log.append(psnr_recon_orig)
                psnr_w_log.append(psnr_recon_w)
                ssim_orig_log.append(ssim_recon_orig)
                ssim_w_log.append(ssim_recon_w)

            # Save the result
            processed_dict = {
                "index": index_log,
                "watermark_gt_str": watermark_gt_str, # Some historical none distructive bug :( will cause this reformatting
                "watermark_decoded": watermark_decoded_log,
                # "mse_orig": mse_orig_log,
                "psnr_orig": psnr_orig_log,
                "ssim_orig": ssim_orig_log,
                # "mse_w": mse_w_log,
                "psnr_w": psnr_w_log,
                "ssim_w": ssim_w_log
            }

            save_name = os.path.join(save_root_dir, file_name)
            with open(save_name, 'wb') as handle:
                pickle.dump(processed_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Decoded Interm. result saved to {}".format(save_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument('--model', type=str, default='saved_models/stegastamp_pretrained')

    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, 
        help="Specification of watermarking method. [rivaGan, dwtDctSvd]",
        default="StegaStamp"
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str, 
        help="Dataset [COCO, DiffusionDB]",
        default="COCO"
    )
    parser.add_argument(
        "--evade_method", dest="evade_method", type=str, help="Specification of evasion method.",
        default="vae"
    )
    parser.add_argument(
        "--arch", dest="arch", type=str, 
        help="""
            Secondary specification of evasion method (if there are other choices).

            Valid values a listed below:
                dip --- ["vanila", "random_projector"],
                vae --- ["cheng2020-anchor", "mbt2018", "bmshj2018-factorized"],
                corrupters --- ["gaussian_blur", "gaussian_noise", "bm3d", "jpeg", "brightness", "contrast"]
                diffuser --- Do not need.
        """,
        default="cheng2020-anchor"
    )
    parser.add_argument(
        "--start", dest="start", type=int, help="Specification of evasion method.",
        default=0
    )
    parser.add_argument(
        "--end", dest="end", type=int, help="Specification of evasion method.",
        default=0
    )
    args = parser.parse_args()
    main(args)
    
    # root_lv1 = os.path.join("Result-Interm", args.watermarker, args.dataset)
    # corrupter_names = [f for f in os.listdir(root_lv1)]
    # for corrupter in corrupter_names:
    #     root_lv2 = os.path.join(root_lv1, corrupter)
    #     arch_names = [f for f in os.listdir(root_lv2)]
    #     for arch in arch_names:
    #         args.evade_method = corrupter
    #         args.arch = arch
    #         print("Processing: {} - {} - {} - {}".format(args.watermarker, args.dataset, args.evade_method, args.arch))
    #         main(args)
    print("\n***** Completed. *****\n")