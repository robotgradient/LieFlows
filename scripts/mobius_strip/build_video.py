import os
from moviepy.editor import *
import numpy as np
from multiprocessing import Process
from liesvf.utils import makedirs
import time

## Parameters ##
base_dir = os.path.abspath(os.path.dirname(__file__))
img_folder = os.path.join(base_dir,'video')
video_save_folder = base_dir
video_name = 'mobius_cycle'

def figure_name(itr):
    return 'mobius_test_'+str(itr)+'.png'


max_figs = 5000
indexes = np.linspace(0,max_figs, max_figs ,dtype='int')

def make_video(video_name='mobius'):
    try:
        clips = []
        for index in indexes:
            try:
                filename = figure_name(index)
                file = os.path.join(img_folder,filename)
                clip = ImageClip(file).set_duration(0.02)
                clips.append(clip)
                #clip.reader.close()
            except:
                pass

        video = concatenate_videoclips(clips, method='compose')
        filename = 'video_{}.mp4'.format(video_name)
        video_save = os.path.join(video_save_folder,filename)
        video.write_videofile(video_save, fps=24)

        for i in range(len(clips)):
            c =  clips.pop()
            c.close()
            del c
        #video.reader.close()
        del clips
        del video

        time.sleep(3)
    except:
        print("BREAK ALL")


p = Process(target=make_video, args=(video_name,))
p.start()
p.join()
