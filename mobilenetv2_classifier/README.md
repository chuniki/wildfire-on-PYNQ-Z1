# Folder Content

- Modules: all codes to train the Pytorch FP32 and the Brevitas quantized models.
- config.py: setup experiments folders, choose Pytorch or Brevitas and setup training hyperparameters and more.
- train.ipynb: trains FP32 model in Pytorch.
- train_brevitas.ipynb: trains Brevitas model.

It includes both yaml files to install all dependencies required in Anaconda.

# How to Use
All the setup must be configured in ```config.py```:
- Logs section: experiments folders for logs, training plots, weights and ONNX export.
- Folders and Datasets: DS_LEN controls the dataset length, to perform quick tests with smaller datasets.
- Hyperparameters and More:
  - BREVITAS_MODEL: choose between FP32 or Quantized model, which is used to decide if some libraries are imported.
  - LOAD_MODEL: allows to load a checkpoint and keep on training.
  - Quantization: configure the bit precision of different stages of the model, to overwrite defaults values.

Finally, to train the models:
- train.ipynb: trains FP32 model in Pytorch.
- train_brevitas.ipynb:
  - Trains Brevitas model.
  - Adds a Bipolar Quantizer at the end of the model, to facilitate deployment in FINN and evaluate again to be sure metrics are the same.


## Datasets and Loss Function
Both datasets, [DFire](https://github.com/gaiasd/DFireDataset) and [FASDD](https://github.com/OyamingO/FASDD), are from object detection domain, so ```dataset_dfire.py``` and ```dataset_dfire.py``` read the class of each bounding box and create the binary multi-label as **[smoke, fire]**.

BCE with Logits is the loss function, applied to the 2 output neurons and summed. Smoke class can be weighted for precision, to avoid false positives:
```python
loss_fn = loss.BCE_LOSS(device=config.DEVICE, smoke_precision_weight=config.SMOKE_PRECISION_WEIGHT)
```

## MobileNetV2 Architecture Modifications
The MobilenetV2 code is based in this [repo](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/models/imagenet/mobilenetv2.py) and located in ```modules/modules.model_mobilenetv2_mini_Resnet.py```. We applied following changes to MobileNetV2_MINI_RESNET class initialization through practical experiments:
- Original:
```python
self.cfgs = [
    # t, c, n, s
    [1,  16, 1, 1],
    [6,  24, 2, 2],
    [6,  32, 3, 2],
    [6,  64, 4, 2],
    [6,  96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]

output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
```
- Ours: minimize inverted residuals and final convolution.
```python
self.cfgs = [
    # t, c, n, s
    [1,  8, 1, 1],
    [2,  16, 2, 2],
    [2,  24, 2, 2],
    [4,  32, 3, 2],
    [2,  64, 2, 1],
]
output_channel = _make_divisible(128 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 128
```

## MobileNetV2: Brevitas Quantization Keys
The trickiest changes are inside class InvertedBlock, due to ADD nodes in Resnets. In order to add two quantized branches and keep the same scale and zero point for both, the quantizer of every block must be returned and used accordingly by following block. It is done with following codes:
```python
class InvertedBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, weight_bit_width, act_bit_width, shared_quant):

        ...

        # If Resnet is added:
        #  - Use the output quantizer of previous block to quantize the output of pw-linear
        #  - Quantize the output of ADD node with QuantIdentity.
        if self.identity:
            self.pw_linear_quant = shared_quant.act_quant
            # Add quantizer only needed if block is a Resnet one
            self.quant_identity_out = QuantIdentity(
                act_quant=CommonIntActQuant,
                bit_width=act_bit_width)
        else:
            self.pw_linear_quant = CommonIntActQuant

      ...

      # Assign the quantizer to the pw-linear convolution
      QuantIdentity(
        act_quant=self.pw_linear_quant,
        bit_width=act_bit_width),

      ...

    # Return the block output quantizer:
    #  - If Resnet is added, it will be the quantizer of the ADD output node.
    #  - If no Resnet is added, returns the quantizer of the pw-linear convolution
    def get_shared_quant(self):
        if self.identity:
            return self.quant_identity_out
        else:
            return self.conv[-1] 
```

