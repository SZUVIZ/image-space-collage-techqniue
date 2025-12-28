import torch
import pydiffvg

# Initialize diffvg
def init_diffvg(device: torch.device,
                use_gpu: bool = torch.cuda.is_available(),
                print_timing: bool = False):
    pydiffvg.set_device(device)
    pydiffvg.set_use_gpu(use_gpu)
    pydiffvg.set_print_timing(print_timing)

# Load simple shape svg
def load_shape_svg(shape_type):
    if shape_type == 'circle':
        points = torch.tensor( [[38.5785, 13.2840],
                                [24.1752, 17.6004],
                                [11.9030, 31.3915],
                                [11.7558, 49.0543],
                                [11.8302, 63.2386],
                                [19.4258, 72.8434],
                                [24.2443, 77.3660],
                                [30.8501, 82.7219],
                                [40.8829, 89.0625],
                                [56.8022, 86.2715],
                                [73.1579, 83.1065],
                                [85.5058, 68.8326],
                                [86.7785, 52.7727],
                                [88.2702, 35.4037],
                                [78.7766, 23.9203],
                                [71.2606, 18.8844],
                                [64.4817, 13.7451],
                                [52.5070,  9.1308]] )
        num_control_points = torch.LongTensor([2] * 6)
        shape = pydiffvg.Path(
                        num_control_points=torch.LongTensor(num_control_points),
                        points=points,
                        stroke_width=torch.tensor(0.0),
                        is_closed=True
                        )
    elif shape_type == "square":
        points = torch.tensor([[10,10],[11,10],[12,10],[13,10],[14,10],[15,10],[16,10],[17,10],[18,10],[19,10],[20,10],[12,10],[21,10],[22,10],[23,10],[80,10],[80,80],[10,80]],dtype=torch.float32)
        num_control_points = torch.LongTensor([0] * 18)
        shape = pydiffvg.Path(
                        num_control_points=torch.LongTensor(num_control_points),
                        points=points,
                        stroke_width=torch.tensor(0.0),
                        is_closed=True
                        )
    return shape

def load_shape_svg_by_photos(photo_list):
    shapes = []
    photo_properties_list = []
    for img in photo_list:
        if img.height>=img.width:
            scale = 300/img.height
            points = torch.tensor([[(300-img.width*scale)/2,0],
                                   [150+img.width*scale/2,0],
                                   [150+img.width*scale/2,300],
                                   [(300-img.width*scale)/2,300]
                                   ])
            photo_properties_list.append(([int((300-img.width*scale)/2),0],[int(150+img.width*scale/2),300],img.width*scale*300))
        elif img.height<img.width:
            scale = 300/img.width
            points = torch.tensor([[0,(300-img.height*scale)/2],
                                   [300,(300-img.height*scale)/2],
                                   [300,150+img.height*scale/2],
                                   [0,150+img.height*scale/2]
                                   ])
            photo_properties_list.append(([0,int((300-img.height*scale)/2)],[300,int(150+img.height*scale/2)],img.height*scale*300))
        num_control_points = torch.LongTensor([0] * 4)
        shape = pydiffvg.Path(
                        num_control_points=torch.LongTensor(num_control_points),
                        points=points,
                        stroke_width=torch.tensor(0.0),
                        is_closed=True
                        )
        shapes.append(shape)
    return shapes,photo_properties_list

def load_shape_svg_by_wordle(points):
    points = torch.tensor(points,dtype=torch.float32)
    num_control_points = torch.LongTensor([0] * 32)
    shape = pydiffvg.Path(
                    num_control_points=torch.LongTensor(num_control_points),
                    points=points,
                    stroke_width=torch.tensor(0.0),
                    is_closed=True
                    )
    return shape