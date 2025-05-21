# 恒星识别系统 (Star Identification System)

[![WakaTime](https://wakatime.com/badge/user/a7b329b7-d489-40d2-9239-8be7cf83b65e/project/018d0c19-921c-4e32-b5ce-f4af890fa9eb.svg)](https://wakatime.com/badge/user/a7b329b7-d489-40d2-9239-8be7cf83b65e/project/018d0c19-921c-4e32-b5ce-f4af890fa9eb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Zhytou/star-identification)](https://github.com/Zhytou/star-identification/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Zhytou/star-identification)](https://github.com/Zhytou/star-identification/network/members)
[![Twitter Share](https://img.shields.io/twitter/url?url=https%3A%2F%2Fgithub.com%2FZhytou%2Fstar-identification&style=social)](https://twitter.com/intent/tweet?text=Check%20out%20this%20awesome%20star%20identification%20system!&url=https%3A%2F%2Fgithub.com%2FZhytou%2Fstar-identification)

- [恒星识别系统 (Star Identification System)](#恒星识别系统-star-identification-system)
  - [📂 项目结构](#-项目结构)
  - [🛠️ 核心功能](#️-核心功能)
    - [🌌 星图仿真](#-星图仿真)
    - [🔍 图像预处理](#-图像预处理)
    - [🎯 星图识别](#-星图识别)
  - [🚀 快速开始](#-快速开始)

## 📂 项目结构

- 星图仿真：simulate.py | view.py
- 星图预处理：denoise.py | detect.py | extract.py
- 特征生成：generate.py | aggregate.py
- 模型相关：train.py | model.py | dataset.py
- 算法测试：test.py | realshot.py | scripts/*.py
- 星表处理：catalogue.py
- 工具函数：utils.py

## 🛠️ 核心功能

### 🌌 星图仿真

**仿真原理**：

1. **恒星筛选**：基于视轴方向与恒星的角距筛选可见恒星。

  $$
    \theta=arccos{\vec{v_{axis}}\cdot\vec{v_{star}}}\le FOV/2
  $$

2. **坐标转换**：
  
  ![celestial_coord_system](imgs/celestial_coord_system.png)

  $$
  \begin{pmatrix}
  x \\
  y \\
  z
  \end{pmatrix} =
  \begin{pmatrix}
  \cos\alpha \cos\delta \\
  \sin\alpha \cos\delta \\
  \sin\delta
  \end{pmatrix}
  $$

  ![sensor_coord_system](imgs/sensor_coord_system.png)

  $$
  M =
  \begin{pmatrix}
  cos\varphi & sin\varphi & 0 \\
  -sin\varphi & cos\varphi & 0 \\
  0 & 0 & 1
  \end{pmatrix}
  \cdot
  \begin{pmatrix}
  cos\varphi & sin\varphi & 0 \\
  -sin\varphi & cos\varphi & 0 \\
  0 & 0 & 1
  \end{pmatrix}
  \cdot
  \begin{pmatrix}
  cos\varphi & sin\varphi & 0 \\
  -sin\varphi & cos\varphi & 0 \\
  0 & 0 & 1
  \end{pmatrix}
  $$

  ![pixel_coord_system](imgs/pixel_coord_system.png)

  $$
  \begin{cases}
  col = \frac{w}{2}+\frac{x}{z}\cdot\frac{f}{d} \\
  row = \frac{h}{2}+\frac{y}{z}\cdot\frac{f}{d}
  \end{cases}
  $$

3. **灰度确定**：基于二维高斯函数的PSF模型。

  $$
    I(x, y) = I_0 \cdot exp^{\frac{(x-x_0)^2+(y-y_0)^2}{2\sigma^2}}
  $$


**运行效果**：

![star_simulator_gui](imgs/star_simulator_gui.png)

### 🔍 图像预处理

### 🎯 星图识别

## 🚀 快速开始

```bash
# clone the repo
git clone https://github.com/Zhytou/star-identification.git  
cd star-identification  

# install all the packages
pip install -r requirements.txt  

# run realshot test
python -m scripts.chapter4_draw
```
