# ObjCAViT: Improving Monocular Depth Estimation Using Natural Language Models And Image-Object Cross-Attention
This is the official implementation of the paper ["ObjCAViT: Improving Monocular Depth Estimation Using Natural Language
Models And Image-Object Cross-Attention", Dylan Auty and Krystian Mikolajczyk (arXiv:2211.17232)](https://arxiv.org/abs/2211.17232)

## Installation
Create a new conda environment from `conda_environment_files/graphbins.yaml`. It will fail to install CLIP, OpenCV, and the NLTK corpuses needed to work, so do the following to fix it:

```
conda activate graphbins
python -m pip install git+https://github.com/openai/CLIP.git
python -m pip install opencv-python
```

Then start python by typing `python` and do the following:
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
exit()
```

## Params files
Params are handled using OmegaConf, and defined via yaml files. The most up-to-date set of parameters and descriptive comments for them can be found in `params/basicParams.yaml`, which is used during development. To define a new experiment, please refer to that file for instructions on what each of the parameters do.

The general structure of the parameters file is (intended to be) modular. Parameters specific to each dataset are in their own sections, and the names of those sections are the same as the strings used to select those datasets. For example, after parsing the file it is put into `args`, so `args.basic.dataset` will contain the name of the dataset (`nyu` or `kitti`) and to access that dataset's settings we use `args.basic[args.basic.dataset].get(f"{dataset_param_to_get}")`.

### Handling obsolete params file formats (what to do if you get a "key not found" error)
During development, some arguments have been moved around - in particular, the section `args.datasets` no longer exists, with its contents being moved into dataset-specific blocks (`args.nyu` and `args.kitti`). To handle this, we use overrides during validation and inference that take the (always the same) dataset parameters from `params/basicParams.yaml` and write them to the relevant sections of the args.

If you run into a problem with a missing key when trying to re-train any experiments, see the function `check_validate_args()` in [misc_utils.py](misc_utils.py) for an example how to handle this programmatically, or modify the params files you're using manually to have the same dataset sections as `basicParams.yaml`.

## Datasets
Preparation of datasets should be done following [the official BTS repo](https://github.com/cleinc/bts) and the Pytorch-specific instructions linked within there.

All datasets are assumed to be located in `./data`, e.g. `./data/nyu` or `./data/kitti`. This can be changed in the parameter yaml file if needed.

## Training
Define a new parameter file, then run:
```python
python main.py -c /path/to/params/file.yaml
```

Results are written to `./runs` by default, and are in Tensorboard format.

For debugging (0 workers for dataloader, running on only one GPU, small number of iterations for both training and testing), the `--debug` command line flag can be optionally added.

## Validation
Use the `-v` flag. Specify either the params file or the automatically-saved `hparams.yaml`:
1. Using regular params file: code will attempt to find the most recently-modified file called `last.ckpt` in the run directory corresponding to the name of the params file (or `args.basic.name` if set in the params file). This is normally fine, but if there are multiple versions of the experiment (i.e. `run_name/version_0`, `run_name/version_1` etc.) then this may not behave as expected.
2. Using auto-saved `hparams.yaml`: Will find the most recently-modified checkpoint called `last.ckpt` within the same directory that the `hparams.yaml` file is located in. File **must** be named `hparams.yaml` for this to work.

Validation mode runs on only one device, with a batch size of 1. It will save a file called `validation_output.txt` in the run directory, containing two sets of metrics: the image-wise running average, following the formulation used in BTS and AdaBins implementations, and the pixelwise total average across the entire validation set. The former is what is reported in the paper, to facilitate comparison with other methods in the literature.

Validation **must** be run with the `--validate` or `-v` flags, and **must** be run with the most recent code. This may mean that checkpoints do not work, due to changes to the code; if this is the case, the checkpoint or params file may need to be changed to permit the newer code to work with the older checkpoints or params files.

We have released the best NYUv2 checkpoint [here](https://github.com/DylanAuty/ObjCAViT/releases/tag/nyu_graphbins_enet-b5_ocv_pos_learned_bbox_wh_emb_128_old_dl_1).
