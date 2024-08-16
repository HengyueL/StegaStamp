import os
import numpy as np
import pandas as pd
import glob
import time
import warnings
import argparse
from encode_image import encode
from decode_image import decode
warnings.filterwarnings('ignore')


def watermark_np_to_str(watermark_np):
    """
        Convert a watermark in np format into a str to display.
    """
    return "".join([str(i) for i in watermark_np.tolist()])


def watermark_str_to_numpy(watermark_str):
    result = [int(i) for i in watermark_str]
    return np.asarray(result)


    
def main(args):
    data_name = args.dataset

    # Clean Img Path
    clean_data_root = os.path.join(
        "..", "DIP_Watermark_Evasion",
        "dataset", "Clean", data_name
    )
    files = [f for f in os.listdir(clean_data_root) if ".png" in f]

    # Save Path
    save_dir = os.path.join(
        "..", "DIP_Watermark_Evasion",
        "dataset", "StegaStamp",data_name
    )
    save_encoder_dir = os.path.join(
        save_dir, "encoder_img"
    )
    os.makedirs(save_encoder_dir, exist_ok=True)

    res_dict = {
        'ImageName': [],
        "Encoder": [],
        'Decoder': []
    }
    save_csv_dir = os.path.join(
        save_dir, "water_mark.csv"
    )
    
    secret_np = np.random.binomial(1, 0.5, 100)

    for img_id, file_name in enumerate(files):
        if img_id > 2:
            break
        clean_img_path = os.path.join(
            clean_data_root, file_name
        )

        # === Get encoder args and Encode Image ===
        args.model = 'saved_models/stegastamp_pretrained'
        args.image = clean_img_path
        args.save_dir = save_encoder_dir
        encode(args, secret_np)
        watermark_str = watermark_np_to_str(secret_np)
        res_dict["ImageName"].append(file_name)
        res_dict["Encoder"].append([watermark_str])


        # ==== Decode Image ===+
        saved_hidden_img_name = file_name.replace(".png", "_hidden.png")
        args.decode_img_dir = os.path.join(
            save_encoder_dir, saved_hidden_img_name
        )
        print("Decode: {}".format(args.decode_img_dir))
        decoded_str = decode(args)
        res_dict["Decoder"].append([decoded_str])


    df = pd.DataFrame(res_dict)
    df.to_csv(save_csv_dir, index=False)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--dataset", dest="dataset", type=str,
        default="COCO"
    )
    args = parser.parse_args()
    main(args)

    print()
    print("Completed.")
   

    