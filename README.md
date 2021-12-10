#Generate Normal Map

Create a normal map by python3 codes

## Required

- Python3
- Scipy
- Numpy
- imageio

## Usage

./generate_normal_map.py input_file 

### Required arguments:

#### input_file            
input image path, output image path will be in the same file location where your input file, and rename it as {file_name}__normal_map.png

## Optional arguments:
defualt setting of **blur** and **detail_scale** for noraml map is 1 and 10. you can modify as you want in generate_normal _ map.py(line 108,109).
**blur** and **detail_scale** are same configurations in Photoshop->filter->3D->Generate Normal Map
        

## Thanks
This project is modified from https://github.com/Mehdi-Antoine/NormalMapGenerator
This project support python3 which https://github.com/Mehdi-Antoine/NormalMapGenerator does not support.

##中文说明
python3 生成 normal map,  默认的设置**blur** 和 **detail_scale**分别是1和10，基本可以满足需求，如果要自己调节，在代码108和109行改即可。和PS的**blur** and **detail_scale** Photoshop->filter->3D->Generate Normal Map设置一样的。


