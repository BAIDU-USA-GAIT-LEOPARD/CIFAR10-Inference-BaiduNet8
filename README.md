# CIFAR10-Inference-BaiduNet8
Training
=========

The model for inference can be first yielded through the following operation:

python train.py

Inference
=========
To obtain the inference time consumption, please download the PyTorch C++ api and extract it in a libtorch directory, and then run the following operations.

mkdir build

cd build

cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch  
-DTorch_DIR=/usr/local/lib/python3.5/dist-packages/torch/share/cmake/Torch ..

make

cd ..

./build/infer

The returned accuracy should be about 94.32%. On a single V100, the average running time is 0.682 milliseconds for each image.
This was run on PyTorch torch-1.0.1.post2-cp35-cp35m-linux_x86_64.whl. 




