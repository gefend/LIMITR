# LIMITR: Leveraging Local Information for Medical Image-Text Representation
LIMITR is a multi-modal representation learning model for chest X-ray images and reports.  
The model is based on a novel alignment scheme between the visual data and the text, which takes into account both local and global information. 
Furthermore, the model integrates domain-specific information of two types -- lateral images and the consistent visual structure of chest images.
Our representation is shown to benefit three types of retrieval tasks: text-image retrieval, class-based retrieval, and phrase-grounding.

> [LIMITR manuscript](https://openaccess.thecvf.com/content/ICCV2023/papers/Dawidowicz_LIMITR_Leveraging_Local_Information_for_Medical_Image-Text_Representation_ICCV_2023_paper.pdf) <br>
> Gefen Dawidowicz, Elad Hirsch, Ayellet Tal <br> 
> Technion – Israel Institute of Technology <br>
> Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023 <br>

![LIMITR](limitr.png)

## Installation
We used Python 3.8 with pytorch 1.11 

To clone this repository:
```shell
git clone https://github.com/gefend/LIMITR.git
```

To install Python requirements:
```shell
pip install -r requirements.txt
```

## Data

1. Download MIMIC-CXR dataset [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).
2. Update the path to MIMIC directory (DATA_BASE_DIR) on `./LIMITR/constants.py`.
3. The splits we used for evaluation and training are available on `./mimic_csv` directory.

## Pretrained Weights

The pretrained model checkpoint and its corresponding config are available on Hugging Face:

| File | Description | Link |
|------|-------------|------|
| `epoch=8-step=44963.ckpt` | Pretrained model checkpoint (frontal, global + local-ext) | [Download](https://huggingface.co/Gefenda/LIMITR/resolve/main/epoch%3D8-step%3D44963.ckpt) |
| `config.yaml` | Corresponding training config | [Download](https://huggingface.co/Gefenda/LIMITR/resolve/main/config.yaml) |

## Training 
Update the desired training configuration on `./configs/mimic_config.yaml`

Train the model with the following command:

```bash
python run.py -c ./configs/mimic_config.yaml --train
```

Test the model with the following command:

```bash
python run.py -c ./configs/mimic_config.yaml --test --ckpt_path=ckpt_path
```
Update `ckpt_path` with the desired checkpoint for evaluation.

## Feature Extraction

LIMITR can extract and save global image and text (report) embeddings for an evaluation dataset using the `extract_features.sh` script.

### Input CSV Format

The input CSV must contain the following columns:

| Column | Description |
|--------|-------------|
| `Path_frontal` | Relative path to the frontal CXR image (relative to `DATA_BASE_DIR` in `constants.py`) |
| `gt` | Ground-truth radiology report text |
| `output` | Generated/predicted report text |


### Running Feature Extraction

The script runs extraction for both `gt` and `output` reports and then merges the per-batch files into final embedding files:

```bash
bash extract_features.sh \
  --ckpt_path /path/to/epoch=8-step=44963.ckpt \
  --features_base_dir /path/to/output/dir \
  --feature_folder my_features \
  --evaldata_csv_path /path/to/evaldata.csv
```

### Output Files

The following merged files are saved to `features_base_dir/feature_folder/`:

| File | Description |
|------|-------------|
| `gt_emb.npy` | Global text embeddings for ground-truth reports |
| `output_emb.npy` | Global text embeddings for generated reports |
| `img_emb.npy` | Global image embeddings |

## Citation
```
@InProceedings{Dawidowicz_2023_ICCV,
    author    = {Dawidowicz, Gefen and Hirsch, Elad and Tal, Ayellet},
    title     = {LIMITR: Leveraging Local Information for Medical Image-Text Representation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {21165-21173}
}
```
