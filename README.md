# Image-Space Collage and Packing with Differential Rendering

### [Project Page](https://szuviz.github.io/pixel-space-collage-technique/)&ensp;&ensp;&ensp;[Paper](https://arxiv.org/pdf/2406.04008)

> Zhenyu Wang,
<a href="https://deardeer.github.io/">Min Lu</a>
<br>
<div>
  <img src="static/teaser_new.png" alt="teaser" width="900" height="auto">
</div>

<!-- > <p>This work presents a novel progressive image vectorization technique aimed at generating layered vectors that represent the original image from coarse to fine detail levels. Our approach introduces semantic simplification, which combines Score Distillation Sampling and semantic segmentation to iteratively simplify the input image. Subsequently, our method optimizes the vector layers for each of the progressively simplified images. Our method provides robust optimization, which avoids local minima and enables adjustable detail levels in the final output. The layered, compact vector representation enhances usability for further editing and modification. 
</p> -->
>
> <p>Collage and packing techniques are widely used to organize geometric shapes into cohesive visual representations, facilitating the representation of visual features holistically, as seen in image collages and word clouds. Traditional methods often rely on object-space optimization, requiring intri- cate geometric descriptors and energy functions to handle complex shapes. In this paper, we introduce a versatile image-space collage technique. Lever- aging a differentiable renderer, our method effectively optimizes the object layout with image-space losses, bringing the benefit of fixed complexity and easy accommodation of various shapes. Applying a hierarchical resolution strategy in image space, our method efficiently optimizes the collage with fast convergence, large coarse steps first and then small precise steps. The diverse visual expressiveness of our approach is demonstrated through various examples. Experimental results show that our method achieves an order of magnitude speedup performance compared to state-of-the-art techniques.
</p>

## Installation
<!-- We suggest users to use the conda for creating new python environment. 

**Requirement**: 5.0<GCC<6.0;  nvcc >10.0. -->

```bash
git clone https://github.com/showery22/pixel-space-collage-technique.git
conda create -n psc python=3.10
conda activate psc
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
```bash
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
conda install -y numpy
conda install -y scikit-image
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite
pip install svgpathtools
pip install cssutils
pip install numba
pip install torch-tools
pip install visdom
python setup.py install
```
<!-- ```bash
cd ..
cd LayeredVectorization
pip install -r requirements.txt
``` -->
<!-- ## Model Checkpoints
- **`SAM`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- **`SD`: "runwayml/stable-diffusion-v1-5"** -->

## Run
```bash
conda activate psc
# Generate the outer contour of the element shape.
python outline_svg.py --config config/config.yaml ----primitive_class any_shape_raster
python main.py --config config/config.yaml --target_shape_mask_dir ./data/target_imgs/s.png \
--shape_class closed --primitive_class any_shape_raster
```

## Reference

    @inproceedings{wang2025image,
      title={Image-Space Collage and Packing with Differentiable Rendering},
      author={Wang, Zhenyu and Lu, Min},
      booktitle={Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
      pages={1--11},
      year={2025}
    }


