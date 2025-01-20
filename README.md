# OP-TR: Improving OPERA by Reimplementing Over-Trust Penalty and Introducing Trust Reward

This repository provides the code implementation of Youlong Ding and Lingyun Xu's MLLM 2025 fianl project, which is an improvement based on the following work: 
> [**OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation**](https://arxiv.org/pdf/2311.17911.pdf) <br>
> [Qidong Huang](https://shikiw.github.io/)<sup>1,2</sup>, 
> [Xiaoyi Dong](https://scholar.google.com/citations?user=FscToE0AAAAJ&hl=en)<sup>2</sup>, 
> [Pan Zhang](https://panzhang0212.github.io/)<sup>2</sup>,
> [Bin Wang](https://wangbindl.github.io/) <sup>2</sup>,
> [Conghui He](https://conghui.github.io/) <sup>2</sup>, 
> [Jiaqi Wang](https://myownskyw7.github.io/)<sup>2</sup>,
> [Dahua Lin](http://dahua.site/)<sup>2</sup>, 
> [Weiming Zhang](http://staff.ustc.edu.cn/~zhangwm/index.html)<sup>1</sup>, 
> [Nenghai Yu](https://scholar.google.com/citations?user=7620QAMAAAAJ&hl=en)<sup>1</sup> <br>
> <sup>1</sup>University of Science and Technology of China, <sup>2</sup>Shanghai AI Laboratory <br>

## Overview
<p align="center"><img src="./teaser.png" alt="teaser" width="500px" /></p>

## Setup
### Environment
```
conda env create -f environment.yml
conda activate opera
```

### Model and Data for Evaluation

The following evaluation requires for MSCOCO 2014 dataset. Please download [here](https://cocodataset.org/#home) and extract it in your data path.

```bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2014.zip
```
Besides, it needs you to prepare the following checkpoints of 7B base models:

- Download [LLaVA-1.5 merged 7B model](https://huggingface.co/liuhaotian/llava-v1.5-7b) and specify it at [Line 14](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/eval_configs/llava-1.5_eval.yaml#L14) of `eval_configs/llava-1.5_eval.yaml`.
```bash
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
```
- Download [Vicuna 7B v1.1 model](https://github.com/lm-sys/FastChat) and specify it at [Line 25](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/minigpt4/configs/models/blip2_instruct_vicuna7b.yaml#L25) of `minigpt4/configs/models/blip2_instruct_vicuna7b.yaml`.
```bash
git clone https://huggingface.co/lmsys/vicuna-7b-v1.1
```
- Download [Vicuna 7B v0 model](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) and specify it at [Line 18](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/minigpt4/configs/models/minigpt4_vicuna0.yaml#L18) of `minigpt4/configs/models/minigpt4_vicuna0.yaml`.
```bash
git clone https://huggingface.co/lmsys/vicuna-7b-delta-v0
```

### Arguments

| Argument             | Example             | Description   |
| -------------------- | ------------------- | ------------- |
| `--model`    | `llava-1.5` | Specify the MLLM model, this codebase supports `instructblip`, `minigpt4`, `llava-1.5`, `shikra`. |
| `--data-path`     | `/path/to/dataset` | Path to the dataset file or folder, e.g., `COCO_2014/val2014/`. |
| `--pope-type`     | `random` | Type for POPE evaluation, supports `random`, `popular`, `adversarial`. |
| `--scale_factor`   | `50` | The scale factor to scale up the self-attention weights. Default: 50. |
| `--threshold`      | `15` | The threshold for attending retrospection. Default: 15. |
| `--num_attn_candidates`   | `5` | The number of candidates per beam. Default: 5. |
| `--penalty_weights`| `1` | The weight of penalty term in decoding. Default: 1.  |

## [Important]Using our released results
We provide our generated output in the [`log`](https://github.com/AvidJoyceXu/OPERA/tree/main/log/) directory. Files in [`log/llava-1.5`](https://github.com/AvidJoyceXu/OPERA/tree/main/log/llava-1.5) are outputs under the `llava-1.5` MLLM models. 

There're two kinds of `jsonl` file. 
- `opera-*` file is for CHAIR ourput generated by the OPERA baseline
- `<INT>.jsonl` file is for CHAIR output generated by our OPTR method, and the correlation between output idx, such as `1.jsonl`, and the exact hyperparameter pair, is specified as follows:

|                  | alpha | $d_0$ | c         | reward |
| ---------------- | ----- | ----- | --------- | ------ |
| 1                | 1.0   | 5     | log 0.1   | log 15 |
| 2                | 2.0   | 7     | log 0.1   | log 15 |
| 3                | 1.0   | 7     | log 0.2   | log 15 |
| 4                | 1.0   | 7     | log 0.1   | log 30 |
| 5                | 1.0   | 7     | log 0.1   | log 7  |
| 6                | 1.0   | 7     | log 0.1   | log 15 |
| 7                | 1.0   | 7     | log 0.5   | log 15 |
| 8                | 0.5   | 7     | log 0.05  | log 15 |
| 9   | 0.8   | 6     | log 0.01  | log 15 |
| 10  | 1.0   | 7     | log 0.05  | log 5  |
| 11               | 0.7   | 5     | log 0.08  | log 20 |
| 12               | 0.8   | 6     | log 0.005 | log 15 |
| 13               | 0.8   | 6     | log 0.01  | log 5  |
| 14               | 1.0   | 7     | log 0.05  | 0      |

## Evaluation Scripts
### CHAIR evaluation
It takes about 2.5h to generate captioning for 500 images on an NVIDIA A100 80GB PCIe acceleration card.

### OP-TR output generation
- Use `Youlong/scripts/llava-run.py` to automatically replace the orginal `transformers-4.29.2/src/transformers/generation/utils.py` file with OP-TR implemented `utils.py` 
  - **NOTICE**: change the [path-related variables](https://github.com/AvidJoyceXu/OPERA/blob/main/Youlong/scripts/llava_run.py#L56-L70) in `llava-run.py` to your own path (to the OPERA base directory, data directory, and the model directory)
  - the set of`Youlong/utils_<INT>.py` is OP-TR implemented `utils.py` with different hyperparameters.

- examine with different hyperparameter combinations:

  - Modify the hyperparameters in the default values of the `generate` member function's parameter in [`utils-<INT>.py`](https://github.com/AvidJoyceXu/OPERA/blob/main/Youlong/utils_1.py#L1166-L1169).
    ```bash
    class GenerationMixin():
        ...
        def generate(
            **args,
            ...
            alpha_d: Optional[float] = 1.0,
            d_0: Optional[int] = 5,
            c_: Optional[float] = math.log(0.1),
            Reward: Optional[float] = math.log(15),
        )
    ```
  - create a new `utils-<INT>.py`, add to the `Youlong` directory, and specify the file name in the [`Youlong/scripts/llava-run.py`](https://github.com/AvidJoyceXu/OPERA/blob/main/Youlong/scripts/llava_run.py#L56-L70) script.
  ```bash

  ```
### OPERA output generation
- Generate the MLLM's responses and save them in a jsonl file:
```bash
python chair_eval.py --model MODEL_NAME --data_path /path/to/COCO --gpu-id GPU_IDs --beam 5 --scale_factor 50 --threshold 15 --num_attn_candidates 5 --penalty_weights 1 --output_path OUTPUT_PATH
```
Note: Please check out our released results in `log/llava-1.5` and `log/instructblip` for reproduction.

- Calculate CHAIR using the generated jsonl file:
```bash
python chair.py --cap_file /path/to/jsonl --image_id_key image_id --caption_key caption --coco_path /path/to/COCO/annotations_trainval2014/annotations/ --save_path /path/to/save/jsonl
```
### POPE evaluation
```bash
python pope_eval.py --model MODEL_NAME --data_path /path/to/COCO --pope-type random --gpu-id GPU_IDs --beam 5 --scale_factor 50 --threshold 15 --num_attn_candidates 5 --penalty_weights 1
```
Result on `Random` split:

| Model | Accuracy | Precision | Recall | F1 score| Yes ratio |
| ----- | -------- | --------- | ------ | ------- | --------- |
| InstructBLIP 7B | 90.3 | 93.8 | 87.0 | 90.3 | 47.8 |
| MiniGPT-4 7B | 79.8 | 89.7 | 68.7 | 77.8 | 39.5 |
| LLaVA-1.5 7B | 89.4 | 90.4 | 88.8 | 89.6 | 50.6 |



