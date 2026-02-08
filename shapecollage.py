from PIL import Image
import copy
import random
import os
import time
from loss import *
from utils.svg_process import init_diffvg,load_shape_svg,load_shape_svg_by_photos,load_shape_svg_by_wordle
import pandas as pd
from tqdm import tqdm
import omegaconf
import nltk
from nltk.corpus import stopwords
from collections import Counter


class ShapeCollage:

    def __init__(self,args: omegaconf.DictConfig,device) -> None:

        self.start_time = time.time()

        self.device = device
        init_diffvg(self.device)

        self.shape_class = args.shape_class

        self.primitive_class = args.primitive_class

        self.weights_list = args.weights_list

        self.primitive_num = len(self.weights_list)

        self.primitive_dir = args.primitive_dir
        self.primitive_mask_list = []
        self.primitive_json_list = []
        self.primitive_outline_list = []
        self.primitive_uniform_list = []
        
        if self.primitive_class not in ["simple_shape","photo_collage","wordle"]:
            self.primitive_json_list = [f"{args.primitive_dir}/outline_files/{i+1}.json" for i in range(self.primitive_num)]
            self.primitive_mask_list = [f"{args.primitive_dir}/outline_files/{i+1}.png" for i in range(self.primitive_num)]
            self.primitive_outline_list = [f"{args.primitive_dir}/outline_files/outline_{i+1}.svg" for i in range(self.primitive_num)]
            if self.primitive_class != "any_shape_raster":
                self.primitive_uniform_list = [f"{args.primitive_dir}/outline_files/uniform_{i+1}.svg" for i in range(self.primitive_num)]
            if self.primitive_class == "any_shape_raster":
                self.primitive_uniform_list = [f"{args.primitive_dir}/outline_files/uniform_{i+1}.png" for i in range(self.primitive_num)]
        

        self.primitive_mask_img_list = []
        self.primitive_area_list = []
        self.centroid_list = []

        self.primitive_pos_init_type=args.primitive_pos_init_type

        self.primitive_fill_ratio = args.primitive_fill_ratio
        self.target_shape_mask_dir = args.target_shape_mask_dir
        self.save_name = args.save_name
        self.shapes = []

        self.base_lr = args.base_lr
        
        self.render_img_size = int(args.render_img_size)
        
        self.num_iters = args.num_iters
        if self.shape_class == "open":
            self.num_iters *= 10
            self.primitive_fill_ratio *= 0.6
        if args.is_global_size:
            self.num_iters *= 5
        
        self.base_lr["size"] *= self.render_img_size/100
        self.base_lr["pos"] *= self.render_img_size/100
        
        
        self.multi_scale_resolution_list = args.multi_res_list
        if self.shape_class == "open":
            self.multi_scale_resolution_list = [1]
        self.is_global_size = args.is_global_size
        

        if self.primitive_class == "wordle":
            self.get_word_list()

    def get_word_list(self):
        if self.word_file.endswith('.txt'):
            with open (self.word_file,'r',encoding='utf-8') as file:
                text = file.read()
                tokens = nltk.word_tokenize(text)
                tokens = [word for word in tokens if word.isalpha()]
                filtered_words = [word for word in tokens if word not in stopwords.words('english')]
                word_frequencies = Counter(filtered_words)
                self.word_list = list(word_frequencies.keys())
                self.weights_list = list(word_frequencies.values())
        elif self.word_file.endswith('.csv'):
            df = pd.read_csv(self.word_file)
            self.word_list = df.iloc[:, 0].tolist()
            self.weights_list = df.iloc[:, 1].tolist()
            self.weights_list = [x**(1/2) for x in self.weights_list]
        else:
            raise Exception("Please enter the correct file format.(.txt or .csv)")
        self.weights_list = self.weights_list[:self.max_words_num]
        self.word_list = self.word_list[:self.max_words_num]
        self.primitive_num = len(self.weights_list)

    def get_raw_control_points(self):
        raw_control_points_list = []
        for i,shape in enumerate(self.shapes):
            control_points = shape.points.to(self.device)
            centroid = torch.tensor(self.centroid_list[i],device=self.device)
            control_points-=centroid
            raw_control_points_list.append(control_points)
        raw_control_points_tensor = torch.stack(raw_control_points_list, dim=0)
        raw_control_points_tensor.requires_grad = False
        return raw_control_points_tensor
    
    def init_shapes_properties(self):
        size_list = []
        angle_list = []
        pos_list = []

        all_area = 0
        for i in range(self.primitive_num):
            all_area+=self.primitive_area_list[i]*self.weights_list[i]*self.weights_list[i]
        background = Image.open(self.target_shape_mask_dir).convert('L').resize((self.render_img_size,self.render_img_size))
        binary_image = np.array(background) < 128
        shape_area = np.sum(binary_image == True)
        global_scale = (self.primitive_fill_ratio*shape_area/all_area)**(1/2)


        if self.primitive_pos_init_type=="distance_field":
            init_pos_list = init_primitive_pos_by_distance_field(self.weights_list,
                                                                self.target_shape_mask_dir,
                                                                primitive_mask_list=self.primitive_mask_img_list,
                                                                centroid_list=self.centroid_list,
                                                                primitive_area_list = self.primitive_area_list,
                                                                global_scale=global_scale,
                                                                img_size=self.render_img_size)


        elif self.primitive_pos_init_type=="random_within_shape":
            init_pos_list = init_primitive_pos_by_random_within_shape(self.weights_list,
                                                                self.target_shape_mask_dir,
                                                                primitive_mask_list=self.primitive_mask_img_list,
                                                                centroid_list=self.centroid_list,
                                                                global_scale=global_scale,
                                                                img_size=self.render_img_size)
        elif self.primitive_pos_init_type=="medial_axis":
            init_pos_list = init_primitive_pos_by_medial_axis(self.target_shape_mask_dir,self.primitive_num,self.render_img_size)

        for i in range(self.primitive_num):
            size = torch.tensor(self.weights_list[i]*global_scale,dtype=torch.float32,device=self.device)
            angle = torch.tensor(0.0,dtype=torch.float32,device=self.device)
            pos = torch.tensor((init_pos_list[i][0],init_pos_list[i][1]),dtype=torch.float32,device=self.device)

            size_list.append(size)
            angle_list.append(angle)
            pos_list.append(pos)
        
        size_list = torch.stack(size_list, dim=0)
        angle_list = torch.stack(angle_list, dim=0)
        pos_list = torch.stack(pos_list, dim=0)

        size_list = size_list.view(self.primitive_num, 1, 1)
        angle_list = angle_list.view(self.primitive_num, 1)
        pos_list = pos_list.view(self.primitive_num,1,2)

        return size_list,angle_list,pos_list

    def init_path_groups(self,is_exclude=False):
        path_groups = []
        for i in range(self.primitive_num):
            if is_exclude:
                path_group = pydiffvg.ShapeGroup(
                                        shape_ids=torch.LongTensor([i]),
                                        fill_color=torch.FloatTensor([0,0,0,0.3]),
                                        stroke_color=torch.FloatTensor([0,0,0,0.3])
                                    )
            else:
                path_group = pydiffvg.ShapeGroup(
                                        shape_ids=torch.LongTensor([i]),
                                        fill_color=torch.FloatTensor([0,0,0,1]),
                                        stroke_color=torch.FloatTensor([0,0,0,1])
                                    )
            path_groups.append(path_group)
        return path_groups

    def get_primitive_mask_img_list(self):
        primitive_mask_img_list = []
        if self.primitive_class == "simple_shape":
            if isinstance(self.shape_type,str):
                if self.shape_type == "circle":
                    image = Image.new('RGBA', (100,100), (0,0,0,0))
                    draw = ImageDraw.Draw(image)
                    draw.ellipse((10, 10, 90, 90), fill=(255,255,255,255))
                elif self.shape_type =="square":
                    image = Image.new('RGBA', (100,100), fill=(255,255,255,255))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle([(10, 10), (90, 90)], fill=(255,255,255,255))
                primitive_mask_img_list = [image]*self.primitive_num
            elif isinstance(self.shape_type,list):
                for i,shape_type in enumerate(self.shape_type):
                    for _ in range(self.shape_num[i]):
                        if shape_type == "circle":
                            image = Image.new('RGBA', (100,100), (0,0,0,0))
                            draw = ImageDraw.Draw(image)
                            draw.ellipse((10, 10, 90, 90), fill=(255,255,255,255))
                        elif shape_type =="square":
                            image = Image.new('RGBA', (100,100), (0,0,0,0))
                            draw = ImageDraw.Draw(image)
                            draw.rectangle([(10, 10), (90, 90)], fill=(255,255,255,255))
                        primitive_mask_img_list.append(image)
        elif self.primitive_class in ["any_shape_svg","any_shape_raster"]:
            for mask_dir in self.primitive_mask_list:
                mask_img = Image.open(mask_dir)
                mask_array = np.array(mask_img.convert("L"))
                binary_image = mask_array>128
                image = np.zeros((mask_img.height, mask_img.width, 4), dtype=np.uint8)
                image[binary_image] = (255,255,255,255)
                image[~binary_image] = [0, 0, 0, 0]
                image = Image.fromarray(image)
                primitive_mask_img_list.append(image)
        elif self.primitive_class == "photo_collage":
            for photo_properties in self.photo_properties_list:
                image = Image.new('RGBA', (300,300), (0,0,0,0))
                draw = ImageDraw.Draw(image)
                draw.rectangle([tuple(photo_properties[0]), tuple(photo_properties[1])], fill=(255,255,255,255))
                primitive_mask_img_list.append(image)
        return primitive_mask_img_list

    def shape_collage(self):
        file_name = str(int(time.time()))
        save_path = f"./workdir/{self.primitive_class}/{file_name}_{self.save_name}"
        os.makedirs(save_path, exist_ok=True)

        if self.primitive_class == "simple_shape":
            if isinstance(self.shape_type,list):
                if not isinstance(self.shape_num,list):
                    raise Exception("The number of shapes does not match the number of weights!")
                if self.primitive_num != sum(self.shape_num):
                    raise Exception("The number of shapes does not match the number of weights!")
                
            if isinstance(self.shape_type,str):
                for _ in range(self.primitive_num):
                    shape = load_shape_svg(self.shape_type)
                    self.shapes.append(shape)
            elif isinstance(self.shape_type,list):
                for i,shape_type in enumerate(self.shape_type):
                    for _ in range(self.shape_num[i]):
                        shape = load_shape_svg(shape_type)
                        self.shapes.append(shape)
            self.centroid_list = [(50,50)]*self.primitive_num
            self.primitive_area_list = [80*80]*self.primitive_num
            self.primitive_mask_img_list = self.get_primitive_mask_img_list()

        elif self.primitive_class in ["any_shape_svg","any_shape_raster"]:
            if self.primitive_num != len(self.primitive_outline_list):
                raise Exception("The number of shapes does not match the number of weights!")
            
            for primitive_outline in self.primitive_outline_list:
                _, _, shapes, _ = pydiffvg.svg_to_scene(primitive_outline)
                self.shapes.append(shapes[0])
            for i in range(self.primitive_num):
                with open(self.primitive_json_list[i], 'r') as file:
                    data = json.load(file)
                    self.centroid_list.append((data["center_x"],data["center_y"]))
                    self.primitive_area_list.append(data["area"])
            self.primitive_mask_img_list = self.get_primitive_mask_img_list()

        elif self.primitive_class == "photo_collage":
            self.photo_list = get_photo_list(self.photos_dir)
            self.shapes,self.photo_properties_list = load_shape_svg_by_photos(self.photo_list)
            self.centroid_list = [(150,150)]*self.primitive_num
            self.primitive_area_list = [x[2] for x in self.photo_properties_list]
            self.primitive_mask_img_list = self.get_primitive_mask_img_list()


        elif self.primitive_class == "wordle":
            for word in self.word_list:
                word_img,mask_img,points,centroid,primitive_area = get_words(word,self.font_path,self.word_color)
                self.primitive_uniform_list.append(word_img)
                self.primitive_mask_img_list.append(mask_img)
                self.shapes.append(load_shape_svg_by_wordle(points))
                self.centroid_list.append(centroid)
                self.primitive_area_list.append(primitive_area)

        raw_control_points_tensor = self.get_raw_control_points()

        size_tensor,angle_tensor,pos_tensor= self.init_shapes_properties()
        shape_groups = self.init_path_groups()
        shape_groups_transparency = self.init_path_groups(is_exclude=True)


        target_area_list = torch.tensor(self.weights_list,device=self.device)
        
        global_size = torch.tensor(1,device=self.device,dtype=torch.float32,requires_grad=False)


        for multiple in self.multi_scale_resolution_list:
            size_tensor,angle_tensor,pos_tensor,global_size = self.mutate_resolution(size_tensor,angle_tensor,pos_tensor,global_size,
                                                                                     raw_control_points_tensor,shape_groups,
                                                                                     shape_groups_transparency,
                                                                                     target_area_list,save_path,multiple)


        if self.primitive_class == "photo_collage":
            post_photos(self.shapes,self.photo_list,self.render_img_size,save_path)
        if self.primitive_class == "any_shape_svg":
            post_svgs(size_tensor,angle_tensor,pos_tensor,global_size,self.primitive_num,self.primitive_uniform_list,self.primitive_json_list,
                      self.is_global_size,save_path)
        if self.primitive_class == "any_shape_raster":
            post_raster(size_tensor,angle_tensor,pos_tensor,global_size,self.primitive_num,self.primitive_uniform_list,self.primitive_mask_list,
                        self.primitive_json_list,self.is_global_size,self.render_img_size,save_path)
        if self.primitive_class == "wordle":
            post_words(size_tensor,angle_tensor,pos_tensor,global_size,self.primitive_num,self.primitive_uniform_list,self.centroid_list,
                       self.is_global_size,self.render_img_size,save_path)
        if self.primitive_class == "simple_shape":
            shape_groups_1 = self.init_path_groups(is_exclude=True)
            pydiffvg.save_svg(f"{save_path}/final.svg",
                                self.render_img_size,
                                self.render_img_size,
                                self.shapes,
                                shape_groups_1)
            

    def mutate_resolution(self,size_tensor,angle_tensor,pos_tensor,global_size,
                            raw_control_points_tensor,
                            shape_groups,
                            shape_groups_transparency,
                            target_area_list,
                            save_path,
                            multiple):
        size_tensor.requires_grad=False
        angle_tensor.requires_grad=False
        pos_tensor.requires_grad=False
        global_size.requires_grad=False

        size_tensor = size_tensor*multiple
        pos_tensor  = pos_tensor*multiple
        global_size = global_size*multiple

        self.base_lr["size"] *= multiple
        self.base_lr["pos"] *= multiple

        self.render_img_size*=multiple

        target_img,mask_img = mask2targetimg(self.target_shape_mask_dir,self.device,self.render_img_size,is_binary=False)

        optimizer = init_optimizer(size_tensor,angle_tensor,pos_tensor,global_size,self.shape_class,self.base_lr,is_global_size=self.is_global_size)

        if self.is_global_size:
            diflation_list = get_diflation_list(self.device)


        with tqdm(total=self.num_iters, desc="Processing value", unit="value") as pbar:
            for epoch in range(self.num_iters):

                if self.is_global_size:
                    primitive_list1 = raw_control_points_tensor*size_tensor*global_size
                else:
                    primitive_list1 = raw_control_points_tensor*size_tensor
                points_2 = torch.zeros_like(primitive_list1,device=self.device)
                points_2[:,:,0] = primitive_list1[:,:,0] * torch.cos(angle_tensor) - primitive_list1[:,:,1] * torch.sin(angle_tensor)
                points_2[:,:,1] = primitive_list1[:,:,0] * torch.sin(angle_tensor) + primitive_list1[:,:,1] * torch.cos(angle_tensor)
                points_2 = points_2+pos_tensor
                for i in range(self.primitive_num):
                    self.shapes[i].points = points_2[i]
                raster_img = svg_to_img(self.shapes,shape_groups,self.render_img_size,self.render_img_size)

                loss_mse = weights_mse(mask_img,raster_img, target_img,scale=5e4)

                # 排斥损失
                raster_img_transparency = svg_to_img(self.shapes,shape_groups_transparency,self.render_img_size,self.render_img_size)
                loss_exclude = exclude_loss(raster_img_transparency,scale=0.5)

                loss_uniform = 0
                loss_force = 0
                if self.shape_class == "closed":
                    if self.is_global_size:
                        loss_uniform = uniform_loss(mask_img,raster_img,diflation_list,scale=5e-6)
    
                elif self.shape_class == "open":
                    loss_uniform = 0
                    loss_force = force_loss(pos_tensor,"down",0.1)


                loss = loss_mse+loss_exclude+loss_uniform+loss_force
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pydiffvg.save_svg(f"{save_path}/{self.render_img_size}_{epoch}.svg",
                                self.render_img_size,
                                self.render_img_size,
                                self.shapes,
                                shape_groups)

                pbar.update(1)

        return size_tensor,angle_tensor,pos_tensor,global_size



def init_optimizer(size_list,angle_list,pos_list,global_size,shape_class,base_lr,is_angle=True,is_global_size=False):
    size_vars = []
    angle_vars = []
    pos_vars=[]
    params = {}
    learnable_params = []

    if shape_class == "closed":
        if is_global_size:
            size_vars.append(global_size)
            global_size.requires_grad=True
            size_list.requires_grad=False
            params['size'] = size_vars
            learnable_params.append({'params': params['size'], 'lr': base_lr["size"], '_id': 0})
        else:
            size_vars.append(size_list)
            size_list.requires_grad=True
            params['size'] = size_vars
            learnable_params.append({'params': params['size'], 'lr': base_lr["size"], '_id': 0})

    if is_angle:
        angle_vars.append(angle_list)
        angle_list.requires_grad = True
        params["angle"] = angle_vars
        # learnable_params.append({'params': params["angle"], 'lr': 0.05, '_id': 1})
        learnable_params.append({'params': params["angle"], 'lr': base_lr["angle"], '_id': 1})

    pos_vars.append(pos_list)
    pos_list.requires_grad=True
    params["pos"] = pos_vars
    learnable_params.append({'params': params["pos"], 'lr': base_lr["pos"], '_id': 2})

    optimizer = torch.optim.Adam(learnable_params, betas=(0.9, 0.9), eps=1e-6)
    return optimizer

def get_diflation_list(device):
    diflation_size_list = [5,11,17,23,29,35]
    diflation_list = []
    for size in diflation_size_list:
        model = Dilation(size).to(device)
        diflation_list.append(model)
    return diflation_list

class Dilation(nn.Module):
    def __init__(self,kernel_size):
        super(Dilation, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.kernel_size, padding=int((self.kernel_size-1)/2), bias=False,padding_mode="replicate")
        nn.init.constant_(self.conv.weight, 1.0)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = x / (self.kernel_size*self.kernel_size)
        x = F.relu(x-0.999)
        x = x*1000
        return x