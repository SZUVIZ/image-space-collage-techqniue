import argparse
import yaml
from shapecollage import ShapeCollage
import torch
import time

def load_config(file_path,args):
    with open(file_path, 'r') as file:
        config =  yaml.safe_load(file)
        for key, value in config.items():
            setattr(args, key, value)
    return args

def main(args,device):
    sc = ShapeCollage(args,device)
    sc.shape_collage()

if __name__ == '__main__':
    old_time = time.time()
    parser = argparse.ArgumentParser(
        description="collage",
    )
    # config
    parser.add_argument("-c", "--config", type=str,default="./config/config.yaml",help="YAML/YML file for configuration.")

    parser.add_argument("-tsmd", "--target_shape_mask_dir", default="./data/target_imgs/s.png", type=str)

    parser.add_argument("-sc", "--shape_class", type=str, choices=["open", "closed"], default="closed",help="Specify the shape class (open, closed).")

    parser.add_argument("-pc", "--primitive_class", type=str, choices=["any_shape_raster","any_shape_svg"], default="any_shape_raster",
                        help="Specify the primitive class (simple_shape, any_shape, photo_collage, wordle).")
    
    parser.add_argument("-pd", "--primitive_dir", type=str,  default="./data/demo_svgs", help="Path to element.")

    parser.add_argument("-sn", "--save_name", type=str, default="s",help="Files save name.")

    parser.add_argument("-ris", "--render_img_size", type=str, default=600,help="resolution") # basic_size 600

    parser.add_argument("-mrl", "--multi_res_list", type=str, default=[1,1,1], help="Resolution multiple.") # [1,2,2]->[basic_size*1,basic_size*1*2,basic_size*1*2*2]

    args = parser.parse_args()
    args = load_config(args.config,args)
    device = torch.device("cuda:0")
    main(args,device)