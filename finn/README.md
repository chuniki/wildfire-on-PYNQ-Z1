# FINN Folder contents:

- Folders:
  - Folding config: final hw config and manual folding config after estimate.
  - ONNX models: model used to feed FINN workflow.

- Notebooks:
  - analysis_hw_reports: analyze all outputs of HW report, after estimate or full build, to understand if it is possible to optimize somehow, assigning resources in a better way.
  - compare_post_synth_resources: compares two different full build projects, to highlight key differences.
  - config.py: setup of folders and experiments names.
  - eval_stitched_ip: rtl simulation of the stitched ONNX model.
  - finn_build_config_pynq_z1: all steps to generate the bitfile and deploy folder of model proposed. There are 3 main options:
    - Estimate: only first estimate.
    - Full Build: tries to build the acceleratos, using automatic folding and FIFOs.
    - Full Build Json: uses a specific folding config file.
  - make_verilog_stitched_testbench: based in templates, generates input and output data and system verilog testbench.
  - my_metrics.py: code to evaluate the ONNX model.
  - prune_custom_multiple_4_8: code to prune a model, based on sparsity.
  - zip_driver: compressed the deploy folder to copy it to the board.
