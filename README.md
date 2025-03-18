<p align="center">

  <h1 align="center">Theoretical design principles of self-supervised denoising networks</h1>
  <p align="center">
    <a href="https://nica.kaist.ac.kr/people">Hayeong Yu</a>*
    ·
    <a href="https://stevejayh.github.io/">Seungjae Han</a>*
    ·
    <a href="https://nica.kaist.ac.kr/people">Young-Gyu Yoon</a>
  </p>
  <p align="center">
    (* equal contribution)
  </p>
  <h2 align="center">WACV 2025 Oral</h2>

  <h3 align="center"><a href="https://openaccess.thecvf.com/content/WACV2025/html/Yu_Design_Principles_of_Multi-Scale_J-Invariant_Networks_for_Self-Supervised_Image_Denoising_WACV_2025_paper.html">Paper</a> </h3>
  <div align="center"></div>
</p>


<p align="center">
  <a href="">
    <img src="https://lh3.googleusercontent.com/oemfhxf5k7KoBW_5-VMMEYykdP71BeaSNyurwJv77rmqAR3rB2SuRAD7WM4BSLHl-lTzbWMndRdaWhyMrlFVIy8QtaxTiO9WV7XrWxIbTbIvRjJ9Wi1_4V2n6oc-yx2RfQ=w1280" alt="Logo" width="95%">
  </a>
</p>

<strong>
<p align="center">
We report the theoretical design principles of self-supervised denoising networks. We show that a U-Net-shaped blind spot network (U-BSN), whose design is derived by following these principles, achieves superior denoising performance at a low computational cost.
</p>
</strong>
<br>


# Installation
Clone the repository and create an anaconda environment using
```
TODO
```

# Dataset
## SIDD dataset
Please download the data from the [SIDD]() 
```
TODO
```

## DND dataset
```
TODO
```

# Training and Evaluation
```
# single-scale training and multi-scale testing on NeRF-synthetic dataset
python scripts/run_nerf_synthetic_stmt.py 

# multi-scale training and multi-scale testing on NeRF-synthetic dataset
python scripts/run_nerf_synthetic_mtmt.py 

# single-scale training and single-scale testing on the mip-nerf 360 dataset
python scripts/run_mipnerf360.py 

# single-scale training and multi-scale testing on the mip-nerf 360 dataset
python scripts/run_mipnerf360_stmt.py 
```

# Citation
If you find our code or paper useful, please cite
```bibtex
@inproceedings{yu2025design,
  title={Design Principles of Multi-Scale J-invariant Networks for Self-Supervised Image Denoising},
  author={Yu, Hayeong and Han, Seungjae and Yoon, Young-Gyu},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={XXXX--XXXX},
  year={2025}
}
```

# Acknowledgements
This project is built upon [AP-BSN](https://github.com/wooseoklee4/AP-BSN). We thank all the authors for their great work and repos. 
