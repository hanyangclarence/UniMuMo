<br/>
<p align="center">
  <h1 align="center">UniMuMo: Unified Text, Music and Motion Generation</h1>
  <p align="center">
    <a href="https://hanyangclarence.github.io/">Han Yang</a>,
    <a href="https://kun-su.netlify.app/">Kun Su</a>,
    <a href="">Yutong Zhang</a>,
    <a href="https://jiabenchen.github.io/">Jiaben Chen</a>,
    <a href="https://scholar.google.com/citations?user=uEpr4C4AAAAJ&hl=en">Kaizhi Qian</a>,
    <a href="">Gaowen Liu</a>,
    <a href="https://people.csail.mit.edu/ganchuang">Chuang Gan</a>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2410.04534">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://hanyangclarence.github.io/unimumo_demo/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>
</p>

---

This is the official repository of **UniMuMo**, a unified music, motion and text generation model. 
In this repository, we present model and data processing code, as well as the model weights.

![](assets/Teaser.png)

---


## Quick Start

### 1. Conda environment
```bash
# clone project   
git clone https://github.com/hanyangclarence/UniMuMo

# create conda environment
cd UniMuMo
conda create -n unimumo python=3.9
conda activate unimumo

# install dependencies
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip install madmom==0.16.1
 ```  

### 2. Download pretrained weight
The weight of UniMuMo consists of three parts: a music VQ-VAE, a motion VQ-VAE and a music-motion LM. 
For inference, please download the unified weight that includes all three parts from [here](https://huggingface.co/ClarenceY/unimumo/blob/main/full.ckpt).
For data preprocessing or training, only one or two parts of them are required for each stage. 
So please download the separate weights from [here](https://huggingface.co/ClarenceY/unimumo/blob/main/separate_model.zip).

After downloaded, please put the weights into folder `pretrained`

### 3. Run the model
For testing the generation results, run the following command:
```bash
python generate.py --help
  --ckpt                The path to the trained model
  
  # about conditioning
  -mu_p --music_path    The path to the music to be conditioned on
  -mo_p --motion_path   The path to the motion to be conditioned on
  -mu_d, --music_description
                        The conditional description for music
  -mo_d, --motion_description
                        The conditional description for motion
  -t, --generation_target {mu,mo,mumo,text}
                        The output format to generate, choosing from music (mu), motion (mo), joint music motion (mumo)
                        and text description (text)
                        
  # about generation settings 
  -gs, --guidance_scale 
                        Guidance scale (Large => better quality and relavancy to text; Small => better diversity)
  --temperature         Temperature for generation
  -d, --duration        Generated music/motion time, default is 10.0
  --seed                Change this value (any integer number) will lead to a different generation result
  -bs, --batch_size     Number of samples to generate for each prompt each time
  --music_meta_dir      The path to music metadata, for loading optional text prompts, default is ./data/music
  -s --save_path        The folder path to save model output
```
Conditions and generation target and be set arbitrarily, for example:
```bash
# generate music and motion without specific conditions
python generate.py --ckpt path_to_weight -t mumo

# generate music and motion with music text description
python generate.py --ckpt path_to_weight -t mumo -mu_d descriptions_for_music

# generate music conditioned on motion and text
python generate.py --ckpt path_to_weight -t mu -mu_d descriptions_for_music -mo_p path_to_motion_condition

# generate music and motion captions
python generate.py --ckpt path_to_weight -t text -mu_p path_to_music_condition -mo_p path_to_motion_condition
```

For loading the model, here is an example:
```python
from unimumo.models import UniMuMo
from unimumo.motion.utils import visualize_music_motion

model = UniMuMo.from_checkpoint('path_to_checkpoint', device='cuda')

waveform_gen, motion_gen = model.generate_music_motion()

visualize_music_motion(waveform_gen, motion_gen['joint'], 'gen_results', model.motion_fps)
```

## Train the Model


### 0. Code Structure
The default training and inference code organizes the data and files into the following structure.
<details>
  <summary>Show Full Tree Structure</summary>
  
  ```bash
UniMuMo_Project  
|   generate.py  
|   README.md
|   requirements.txt
|   train.py
|
+---assets
|
+---configs  # all configurations and hyperparameters for the three training stage
|       train_caption.yaml
|       train_motion_vqvae.yaml
|       train_music_motion.yaml
|
+---data  # store the training data and metadata
|   +---motion
|   |   |   aist_test.txt  # dataset split for all three motion datasets
|   |   |   aist_train.txt
|   |   |   aist_val.txt
|   |   |   dancedb_test.txt
|   |   |   dancedb_train.txt
|   |   |   dancedb_val.txt
|   |   |   humanml3d_test.txt
|   |   |   humanml3d_train.txt
|   |   |   humanml3d_val.txt
|   |   |   ignore_list.txt
|   |   |   Mean.npy  # mean and std calculated on the three datasets
|   |   |   Std.npy
|   |   |   test_length.pickle  # motion sequence length
|   |   |   train_length.pickle
|   |   |   val_length.pickle
|   |   |
|   |   +---aligned_motion_code  # the folder for all extracted motion codes that are aligned with music, generated by preprocessing/get_aligned_motion_code.py
|   |   +---humanml3d_text_description  # the folder for all HumanML3D text description txt files     
|   |   +---test  # all motion features of shape (T, 263). Train, test and val folder have the same structure
|   |   |   \---joint_vecs                   
|   |   +---train
|   |   \---val
|   |
|   \---music
|       |   music4all_captions_gpt.json  # music captions generated by ChatGPT and Mu-LLaMa
|       |   music4all_captions_mullama.json
|       |   music4all_captions_mullama_val_test.json
|       |   music4all_ignore.txt
|       |   music4all_metadata.csv  # the metadata modified from music4all dataset
|       |   music4all_test.txt  # our split for music4all
|       |   music4all_train.txt
|       |   music4all_val.txt
|       |   musiccaps-public.csv  # the downloaded musiccaps test data
|       |
|       +---audios  # the folder for all music4all .mp3 or .wav files
|       +---music_beat  # the folder for detected music beat, generated by preprocessing/extract_music_code_beat.py
|       \---music_codes  # the folder for extracted music code, generated by preprocessing/extract_music_code_beat.py
|
+---preprocessing 
|       extract_music_code_beat.py  # extract music code with Encodec and detect music beat
|       get_aligned_motion_code.py  # align each music track with several motion sequences
|       get_text_prompt.py  # get music captions from ChatGPT
|
+---test_model
|       demo_motiontext2music.py
|       demo_musictext2motion.py
|       demo_music_motion_alignment_60hz.py
|       demo_t2mm.py
|       test_motion_vqvae.py
|       test_motion2music_aist.py
|       test_motion2text_pad.py
|       test_music2motion_aist.py
|       test_music2text_mullama.py
|       test_musiccaps.py
|
\---unimumo  # the main code for UniMuMo
  ```
</details>


### 1. Prepare the Datasets
#### 1.1 Music dataset
Please refer to the website of [Music4All](https://sites.google.com/view/contact4music4all) to download the dataset. 
After downloaded, put the audio files in folder `data/music/audios`.
#### 1.2 Motion dataset
We use motion data from [HumanML3D](https://github.com/EricGuo5513/HumanML3D), [AIST++](https://google.github.io/aistplusplus_dataset/factsfigures.html) and [DanceDB](https://dancedb.eu/) together, preprocessing them into the same format. For convenience, please directly download the preprocessed motion data from [here](https://huggingface.co/ClarenceY/unimumo/blob/main/motion_data.tar.gz).



### 2. Preprocess the Data
#### 2.1 Split vocals from music (optional)
We use [Demucs](https://github.com/facebookresearch/demucs) for splitting music and vocal.
#### 2.2 Music code extraction and beat detection
To speed up training, we use [Encodec](https://github.com/facebookresearch/audiocraft/blob/main/docs/ENCODEC.md) to 
extract all the music codes and use [drum-aware4beat](https://github.com/SunnyCYC/drum-aware4beat) to track 
the music beat before training. Please set the correct data path in `preprocessing/extract_music_code_beat.py` and run:
```bash
python preprocessing/extract_music_code_beat.py --start 0.0 --end 1.0
```
Since this process takes a long time, if you have multiple machines, you can split the work by setting `--start` and 
`--end` to specify the start and end point of each job.



### 3. Train Motion VQ-VAE
Please first check the settings in `configs/train_motion_vqvae.yaml`, e.g., the paths of datasets, number of device and node.
Then run:
```bash
python train.py --stage train_vqvae --base configs/train_motion_vqvae.yaml
```
Resuming training can be achieved by appending `-r path_to_previous_checkpoint` to above command.



### 4. Pair Music with Motion and Extract Motion Code
After training the motion VQ-VAE, we use Dynamic Time Warping to pair each music track with several motions and 
extract the motion codes from the augmented motion sequences prior to training the music-motion LM. Please first set the
correct data paths and run:
```bash
python preprocessing/get_aligned_motion_code.py --start 0.0 --end 1.0
```
You can also set `--start` and `--end` to manually distribute the work.



### 5. Train Music-Motion LM
Please first check the settings in `configs/train_lm.yaml`, and run:
```bash
python train.py --stage train_music_motion --base configs/train_lm.yaml
```
Similarly, training can be resumed by appending `-r path_to_previous_checkpoint`.



### 6. Train Captioning Model
Please run:
```bash
python train.py --stage train_caption --mm_ckpt path_to_last_stage_model --base configs/train_lm.yaml
```
Note that it is required to provide the checkpoint of previous stage in `--mm_ckpt`, since the captioning model is built on
the trained music-motion LM.



### 7. Integrate the Trained Weights
Finally, we have three separate model checkpoints: an Encodec, a motion VQ-VAE and a music-motion LM. We combine them into
a single checkpoint that can be directly loaded by `class UniMuMo` by running:
```bash
python unimumo/merge_model_checkpoints.py (provide the paths for all the checkpoints, configs and metadata...)
```



 

## Evaluate the Model

### 1. Run Tasks
All the scripts for testing the model in large-scale are in `test_model` folder, start with "test_". The name of each script signifies the task it works on.

For example, to test the reconstruction loss of the motion VQ-VAE, run
```bash
python test_model/test_motion_vqvae.py --save_dir path_to_save_destination --ckpt path_to_model_ckpt 
```
To test the model on MusicCaps, run
```bash
python test_model/test_musiccaps.py --save_dir path_to_save_destination --ckpt path_to_model_ckpt 
```
and you can also set `start` and `end` config to split the job.


### 2. Quantitative Evaluation
As described in the paper in detail, we directly adopt the evaluation metrics from various repos. Please refer to the paper for further guide on running each metrics.



## Acknowledgement
Our code is partially built on the following repositories: [Audiocraft](https://github.com/facebookresearch/audiocraft),
[Stable Diffusion](https://github.com/CompVis/stable-diffusion), [drum-aware4beat](https://github.com/SunnyCYC/drum-aware4beat)
and [T2M-GPT](https://github.com/Mael-zys/T2M-GPT). Thanks to their great work!



 
