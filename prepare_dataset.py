import os
import numpy as np
import pandas as pd
import glob
import time
import warnings
import argparse
warnings.filterwarnings('ignore')


def watermark_np_to_str(watermark_np):
    """
        Convert a watermark in np format into a str to display.
    """
    return "".join([str(i) for i in watermark_np.tolist()])


def watermark_str_to_numpy(watermark_str):
    result = [int(i) for i in watermark_str]
    return np.asarray(result)

def watermark_str_to_bitstr(watermark_str):
        binary_str = ""
        for letter in watermark_str:
            binary_str += int_to_8_bit_binary_str(ord(letter))
        return binary_str

def to_ascii(str_8_bits):
    power = 7
    res = 0
    for i in str_8_bits:
        res += int(i) * (2 ** power)
        power -= 1
    return res


def int_to_8_bit_binary_str(number):
    return '{0:08b}'.format(number)


def binary_literal_to_str(watermark_binary_literal):
    num_chunks = len(watermark_binary_literal) // 8

    return_str = ""
    for i_chunk in range(num_chunks):
        str_slice = watermark_binary_literal[8*i_chunk:8*(i_chunk+1)]
        ascii_code = int(to_ascii(str_slice))
        letter = chr(ascii_code)
        return_str += letter
    return return_str






def watermark(save_dir_encoder, img_name, original_img_path, hiden_message):
    img_name = img_name.split('.')[0] + '_hidden.png'
    save_encoder_img_name = os.path.join(save_dir_encoder, img_name)
    pretrained_model = 'saved_models/stegastamp_pretrained'
    # for Encoder
    print('\n\n#############################\n\n')
    print('Starting Encoding...')
    cmd = 'python encode_image.py {} --image {} --save_dir {} --secret {}'.format(pretrained_model, original_img_path, save_dir_encoder, hiden_message)
    #os.system(cmd)
    
    
    print('\n\n#############################\n\n')
    os.system('clear')
    #cmd = 'python decode_image.py {} --image {}'.format(pretrained_model, save_encoder_img_name)
    cmd = 'python decode_image.py {} --image {}>temp.txt'.format(pretrained_model, original_img_path)
    os.system(cmd)
    print('\n\n#############################\n\n')
    print('Finishing Decoding...')


def main(args):
    data_name = args.dataset

    result_dict = {
        'ImageName': [],
        'Decoder': []
    }
    watermark_gt = np.random.binomial(1, 0.5, 7)

    # === convert from np to str ===
    watermark_str = watermark_np_to_str(watermark_gt)
    print("Watermark String: ", watermark_str)

    # # === convert from np to binary ===
    watermark_utf = watermark_str.encode("utf-8")
    print("Watermark machine code: ", watermark_utf)

    hiden_message = 'abcd'
    
    count = 0 

    save_dir_encoder = os.path.join(
        "dataset_processed", "StegaStamp",
        data_name
    )
    for img_id in range(1, 2001):
        img_name = "Img-{}.png".format(img_id)
        original_img_path = os.path.join(
            "dataset_clean", "Clean",
            data_name, img_name
        )
        # original_img_path = '/home/jusun/shared/Robust_DL/0_WaterMark/dataset/Clean/{}/Img-{}.png'.format(data_name,img_id)

        count += 1
        
        save_encoder_img_name = os.path.join(save_dir_encoder, img_name)
        watermark(save_dir_encoder, img_name, original_img_path, hiden_message)
        encoder_data = watermark_str_to_bitstr(hiden_message)
        #exit()
        if not os.path.exists('temp.txt'):
            assert False, 'No temp.txt'
        else:
            with open('temp.txt', 'r') as file:
                for line in file:
                    line = line.strip()
                    result_dict['ImageNmae'].append(img_name)
                    #result_dict['Encoder'].append([encoder_data])
                    result_dict['Decoder'].append([line])
                    # code_match = False
                    # if encoder_data == line:
                    #     code_match = True
                    # result_dict['Match'].append(code_match)
            try:
                os.remove('temp.txt')
                os.system('clear')
                #time.sleep(5)
            except:
                pass
                #assert False, 'Remove Error!'
    
    result_df = pd.DataFrame.from_dict(result_dict)
    save_clean_dir = '../Clean_Watermark_Evasion/StegaStamp/{}'.format(data_name)
    if not os.path.exists(save_clean_dir):
        os.makedirs(save_clean_dir)
    
    result_file = '../Clean_Watermark_Evasion/StegaStamp/{}/water_mark.csv'.format(data_name)
    result_df.to_csv(result_file, index=False)


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
   

    