# wildfire-on-PYNQ-Z1

Project to classify wildfires with a MobileNetV2 based network, very reduced.

## Project Organization

- mobilenetv2_classifier: contains all files to train a tiny MobileNetV2 network, both the FP32 model in Pytorch and the quantized version in Brevitas. These are the datasets used:
    - [DFire](https://github.com/gaiasd/DFireDataset).
    - [FASDD](https://github.com/OyamingO/FASDD).
- finn: notebooks to build the driver, analyze results, evaluate the stitched version, create the verilog testbench and prune the ONNX model based on convolution channel sparsity.
- pynq-z1: notebooks to evaluate the datasets and run inference loop with threads and a camera.

