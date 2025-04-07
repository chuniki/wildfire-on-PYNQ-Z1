This project is built with PYNQ-Z1 image 3.0.1.

## Folders Content

- bitfile: it contains the files with the accelerator configuration for the FPGA.
- driver: it contains all the files needed to evaluate the model and run an inference loop with a USB camera.
  - camera_inference_threads.ipynb: runs inference loop with a USB camera, dividing the task in threads and plotting images with asyncio.
  - driver.py: it contains the dictionary (io_shape_dict) with all accelerator input and output features needed to run it.
  - driver_base.py: class to run the driver. It contains two modified functions to run the driver asynchronously and poll the DMA to retrieve the results. They are explained in camera_inference_threads.ipynb.
  - eval_dataset.ipynb: evaluate all images in test datasets, using my_metrics.py.
  - my_metrics.py: compute accuracy, precision, recall and f1.

### Additional Libraries Installation

- bitstring:
```
sudo python -m pip install bitstring
```

- jcv2: it allows drawing images inside jupyter notebooks.
```
sudo python -m pip install opencv-jupyter-ui
```


