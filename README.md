# wildfire-on-PYNQ-Z1

Project to classify wildfires with a tiny MobileNetV2-based network.

## Project Organization

- mobilenetv2_classifier: contains all files to train a tiny MobileNetV2 network, both the FP32 model in Pytorch and the quantized version in Brevitas. These are the datasets used:
    - [DFire](https://github.com/gaiasd/DFireDataset).
    - [FASDD](https://github.com/OyamingO/FASDD).
- finn: notebooks to build the driver, analyze results, evaluate the stitched version, create the verilog testbench and prune the ONNX model based on convolution channel sparsity.
- pynq-z1: notebooks to evaluate the datasets and run inference loop with threads and a camera.

## Metrics

Comparison between the original model and the tiny MobileNetV2 model deployed on the FPGA. 

| Model  | Weights | MAC | F1-Macro |
| :---         |     :---:      |     :---:      |          ---: |
| MobileNetV2 Original  | 2.22 M  | 300 M | 97.65 % |
| MobileNetV2 Nano FPGA  | 68.8 K  | 42 M | 95.32 % |

The tiny MobileNetV2 model model achieves 32 fps with the FPGA clocked at 4 MHz and an average power consumption of 2.5W on the PYNQ-Z1.

| FPGA clock | PYNQ-Z1 average power | fps  |
|     :---:      |     :---:      |     :---:      |
| 4 MHz  | 2.5 W  | 32 | 

