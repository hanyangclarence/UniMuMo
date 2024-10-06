import argparse
import os
import torch
import numpy as np
import random
import librosa
import soundfile as sf
from pytorch_lightning import seed_everything

from unimumo.models import UniMuMo
from unimumo.util import get_music_motion_prompt_list, interpolate_to_60fps
from unimumo.motion.utils import visualize_music_motion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # about loading and saving paths
    parser.add_argument(
        "--save_path",
        type=str,
        required=False,
        help="The path to save model output",
        default="./gen_results",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=False,
        default=None,
        help="The path to the trained model",
    )
    parser.add_argument(
        "--music_meta_dir",
        type=str,
        required=False,
        help="The path to music metadata dir",
        default="../My_Project/data/music",
    )

    # about input prompt
    parser.add_argument(
        "-mu_p",
        "--music_path",
        type=str,
        required=False,
        default=None,
        help="The path to the music to be conditioned on",
    )
    parser.add_argument(
        "-mo_p",
        "--motion_path",
        type=str,
        required=False,
        default=None,
        help="The path to the motion to be conditioned on",
    )
    parser.add_argument(
        "-mu_d",
        "--music_description",
        type=str,
        required=False,
        default=None,
        help="The conditional description of music",
    )
    parser.add_argument(
        "-mo_d",
        "--motion_description",
        type=str,
        required=False,
        default=None,
        help="The conditional description of motion",
    )
    parser.add_argument(
        "-t",
        "--generation_target",
        type=str,
        required=True,
        choices=['mu', 'mo', 'mumo', 'text'],
        help="The output format to generate",
    )

    # about generation settings
    parser.add_argument(
        "-gs",
        "--guidance_scale",
        type=float,
        required=False,
        default=4.0,
        help="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=None,
        help="Temperature for generation",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        required=False,
        default=10.0,
        help="Generated music/motion time",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Change this value (any integer number) will lead to a different generation result.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        required=False,
        default=1,
        help="Number of samples to generate for each prompt.",
    )

    args = parser.parse_args()

    # sanity check of the arguments
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    model_ckpt = args.ckpt
    assert os.path.exists(model_ckpt)
    music_meta_dir = args.music_meta_dir
    assert os.path.exists(music_meta_dir)
    guidance_scale = args.guidance_scale
    batch_size = args.batch_size
    temperature = args.temperature
    duration = args.duration
    seed_everything(args.seed)
    music_path = args.music_path
    motion_path = args.motion_path
    music_description = args.music_description
    motion_description = args.motion_description
    generation_target = args.generation_target

    # currently unconditional generation still not works well, so if description is not provided,
    # we randomly load prompts from our datasets
    music_prompt_list, motion_prompt_list = get_music_motion_prompt_list(music_meta_dir)
    text_description_list = []
    for _ in range(batch_size):
        if music_description is None and motion_description is None:
            music_desc_input = random.choice(music_prompt_list)
            motion_desc_input = random.choice(motion_prompt_list)
        elif music_description is None:
            music_desc_input = random.choice(music_prompt_list)
            motion_desc_input = motion_description
        elif motion_description is None:
            music_desc_input = music_description
            motion_desc_input = random.choice(motion_prompt_list)
        else:
            music_desc_input = music_description
            motion_desc_input = motion_description
        text_description_list.append(music_desc_input.capitalize() + ' <separation> ' + motion_desc_input.capitalize())

    # load model
    model = UniMuMo.from_checkpoint(model_ckpt)

    if generation_target == 'mumo':
        waveform_gen, motion_gen = model.generate_music_motion(
            text_description=text_description_list,
            duration=duration,
            conditional_guidance_scale=guidance_scale,
            temperature=temperature
        )
        waveform_to_visualize = waveform_gen
        motion_to_visualize = motion_gen['joint']
        print(f'waveform_gen: {waveform_to_visualize.shape}, joint: {motion_to_visualize.shape}, text: {text_description_list}')

    elif generation_target == 'mu':
        assert os.path.exists(motion_path), 'When generating motion-to-music, motion path should be provided'

        motion = np.load(motion_path)  # by default the motion is from aist, so the fps should be 60
        if model.motion_fps == 20:  # down sample if the model is trained on 20 fps motion data
            motion = motion[::3]
        motion = motion[None, ...]  # [1, fps * duration, D]
        # cut motion to generate duration
        motion = motion[:, :int(model.motion_fps * duration)]

        waveform_gen = model.generate_music_from_motion(
            motion_feature=motion,
            text_description=text_description_list,
            batch_size=batch_size,
            conditional_guidance_scale=guidance_scale,
            temperature=temperature
        )
        waveform_to_visualize = waveform_gen
        motion_to_visualize = model.motion_vec_to_joint(
            torch.Tensor(model.normalize_motion(motion))
        )
        motion_to_visualize = np.tile(motion_to_visualize, (batch_size, 1, 1, 1))
        print(f'waveform_gen: {waveform_to_visualize.shape}, joint: {motion_to_visualize.shape}, text: {text_description_list}')

    elif generation_target == 'mo':
        assert os.path.exists(music_path), 'When generating music-to-motion, music path should be provided'

        waveform, _ = librosa.load(music_path, sr=32000)
        # cut waveform to generate duration
        waveform = waveform[:int(32000 * duration)]
        waveform = waveform[None, None, ...]  # [1, 1, 32000 * duration]

        motion_gen = model.generate_motion_from_music(
            waveform=waveform,
            text_description=text_description_list,
            batch_size=batch_size,
            conditional_guidance_scale=guidance_scale,
            temperature=temperature
        )
        waveform_to_visualize = np.tile(waveform, (batch_size, 1, 1))
        motion_to_visualize = motion_gen['joint']
        print(f'waveform_gen: {waveform_to_visualize.shape}, joint: {motion_to_visualize.shape}, text: {text_description_list}')

    else:  # generate text
        if motion_path is not None and os.path.exists(motion_path):
            # load motion
            motion_feature = np.load(motion_path)

            if motion_path.split('/')[-1][0] == 'g':  # the motion file is from AIST++ (60 fps)
                if model.motion_fps == 20:
                    motion_feature = motion_feature[::3]
            else:  # by default they are from HumanML3D (20 fps)
                if model.motion_fps == 60:
                    motion_feature = interpolate_to_60fps(motion_feature)

            target_length = int(duration * model.motion_fps)
            curr_lenth = motion_feature.shape[0]
            if curr_lenth >= target_length:
                motion_feature = motion_feature[:target_length]
            else:
                padded_feature = np.zeros((target_length, motion_feature.shape[-1]))
                padded_feature[:curr_lenth] = motion_feature
                motion_feature = padded_feature
            
            motion_feature = motion_feature[None, ...]  # [1, fps * duration, D]
            print(f"motion feature: {motion_feature.shape}")
            
            captions = model.generate_text(motion_feature=motion_feature)
            description = captions[0]
            
            print(f'Generated motion caption: {description}')
            
            waveform_to_visualize = np.zeros((1, 1, int(32000 * duration)))
            motion_to_visualize = model.motion_vec_to_joint(
                torch.Tensor(model.normalize_motion(motion_feature))
            )
            
            # visualize motion
            visualize_music_motion(
                waveform=waveform_to_visualize, joint=motion_to_visualize, save_dir=save_path, fps=model.motion_fps
            )
            
        
        if music_path is not None and os.path.exists(music_path):
            # load music
            waveform, _ = librosa.load(music_path, sr=32000)
            
            # cut waveform to generate duration
            waveform = waveform[:int(32000 * duration)]
            waveform = waveform[None, None, ...]  # [1, 1, 32000 * duration]
            
            print(f"waveform: {waveform.shape}")

            captions = model.generate_text(waveform=waveform)
            description = captions[0]
            
            print(f'Generated music caption: {description}')
            
            # save the audio into the save dir
            sf.write(os.path.join(save_path, 'audio_to_annotate.wav'), waveform.squeeze(), 32000)

            

    
