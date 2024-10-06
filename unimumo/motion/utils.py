import numpy as np
import soundfile
import os
from os.path import join as pjoin
import subprocess

from unimumo.motion import skel_animation


kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                   [9, 13, 16, 18, 20]]


def visualize_music_motion(waveform: np.ndarray, joint: np.ndarray, save_dir: str, fps: int):
    if waveform.ndim == 3:
        waveform = waveform.squeeze(1)
    assert waveform.ndim == 2, 'waveform should be of shape [b, 32000 * duration]'
    assert joint.ndim == 4 and joint.shape[-1] == 3 and joint.shape[-2] == 22, \
        'joint should be of shape [b, 20 * duration, 22, 3]'
    assert fps in [20, 60]
    os.makedirs(save_dir, exist_ok=True)

    batch_size = waveform.shape[0]
    music_path = None
    motion_path = None
    for i in range(batch_size):
        music_path = pjoin(save_dir, 'music.mp3')
        try:
            soundfile.write(music_path, waveform[i], samplerate=32000)
        except Exception as e:
            print(e)
            continue

        motion_path = pjoin(save_dir, 'motion.mp4')
        try:
            skel_animation.plot_3d_motion(
                motion_path, kinematic_chain, joint[i], title='Music-Motion', vbeat=None,
                fps=fps, radius=4
            )
        except Exception as e:
            print(e)
            continue

        video_path = pjoin(save_dir, f'video_{i}.mp4')
        try:
            subprocess.call(
                f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}",
                shell=True)
        except Exception as e:
            print(e)
            continue

    # remove the separate music and motion file
    os.system(f'rm {music_path}')
    os.system(f'rm {motion_path}')
