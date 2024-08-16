from pathlib import Path
from typing import List, Union
import random
import re
from datetime import timedelta

from tqdm import trange
import numpy as np
import librosa
import cv2
from zhon.hanzi import punctuation as zh_punc
from moviepy.editor import ImageClip, AudioFileClip, CompositeAudioClip, \
    CompositeVideoClip, ColorClip, VideoFileClip, VideoClip, TextClip, concatenate_audioclips 
import moviepy.video.compositing.transitions as transfx
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.audio.fx.all import audio_loop
from moviepy.video.tools.subtitles import SubtitlesClip

from mm_story_agent.base import register_tool


def generate_srt(timestamps: List,
                 captions: List,
                 save_path: Union[str, Path],
                 max_single_length: int = 30):
    
    def format_time(seconds: float) -> str:
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        millis = int((td.total_seconds() - total_seconds) * 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"
    
    srt_content = []
    num_caps = len(timestamps)

    for idx in range(num_caps):
        start_time, end_time = timestamps[idx]
        caption_chunks = split_caption(captions[idx], max_single_length).split("\n")
        num_chunks = len(caption_chunks)
        
        if num_chunks == 0:
            continue

        segment_duration = (end_time - start_time) / num_chunks

        for chunk_idx, chunk in enumerate(caption_chunks):
            chunk_start_time = start_time + segment_duration * chunk_idx
            chunk_end_time = start_time + segment_duration * (chunk_idx + 1)
            start_time_str = format_time(chunk_start_time)
            end_time_str = format_time(chunk_end_time)
            srt_content.append(f"{len(srt_content) // 2 + 1}\n{start_time_str} --> {end_time_str}\n{chunk}\n\n")

    with open(save_path, 'w') as srt_file:
        srt_file.writelines(srt_content)


def add_caption(captions: List,
                srt_path: Union[str, Path],
                timestamps: List,
                video_clip: VideoClip,
                max_single_length: int = 30,
                **caption_config):
    generate_srt(timestamps, captions, srt_path, max_single_length)

    generator = lambda txt: TextClip(txt, **caption_config)
    subtitles = SubtitlesClip(srt_path.__str__(), generator)
    captioned_clip = CompositeVideoClip([video_clip,
                                         subtitles.set_position(("center", "bottom"), relative=True)])
    return captioned_clip


def split_keep_separator(text, separator):
    pattern = f'([{re.escape(separator)}])'
    pieces = re.split(pattern, text)
    return pieces


def split_caption(caption, max_length=30):
    lines = []
    if ord(caption[0]) >= ord("a") and ord(caption[0]) <= ord("z") or ord(caption[0]) >= ord("A") and ord(caption[0]) <= ord("Z"):
        words = caption.split(" ")
        current_words = []
        for word in words:
            if len(" ".join(current_words + [word])) <= max_length:
                current_words += [word]
            else:
                if current_words:
                    lines.append(" ".join(current_words))
                    current_words = []

        if current_words:
            lines.append(" ".join(current_words))
    else:
        sentences = split_keep_separator(caption, zh_punc)
        current_line = ""
        for sentence in sentences:
            if len(current_line + sentence) <= max_length:
                current_line += sentence
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = ""
                if sentence.startswith(tuple(zh_punc)):
                    if lines:
                        lines[-1] += sentence[0]
                    current_line = sentence[1:]
                else:
                    current_line = sentence

        if current_line:
            lines.append(current_line.strip())

    return '\n'.join(lines)


def add_bottom_black_area(clip: VideoFileClip,
                          black_area_height: int = 64):
    """
    Add a black area at the bottom of the video clip (for captions).

    Args:
        clip (VideoFileClip): Video clip to be processed.
        black_area_height (int): Height of the black area.

    Returns:
        VideoFileClip: Processed video clip.
    """
    black_bar = ColorClip(size=(clip.w, black_area_height), color=(0, 0, 0), duration=clip.duration)
    extended_clip = CompositeVideoClip([clip, black_bar.set_position(("center", "bottom"))])
    return extended_clip


def add_zoom_effect(clip, speed=1.0, mode='in', position='center'):
    fps = clip.fps
    duration = clip.duration
    total_frames = int(duration * fps)
    def main(getframe, t):
        frame = getframe(t)
        h, w = frame.shape[: 2]
        i = t * fps
        if mode == 'out':
            i = total_frames - i
        zoom = 1 + (i * ((0.1 * speed) / total_frames))
        positions = {'center':  [(w - (w * zoom)) / 2,  (h - (h  *  zoom)) / 2],
                     'left': [0, (h - (h * zoom)) / 2],
                     'right': [(w - (w * zoom)), (h - (h * zoom)) / 2],
                     'top': [(w - (w * zoom)) / 2, 0],
                     'topleft': [0, 0],
                     'topright': [(w - (w * zoom)), 0],
                     'bottom': [(w - (w * zoom)) / 2, (h - (h * zoom))],
                     'bottomleft': [0, (h - (h * zoom))],
                     'bottomright': [(w - (w * zoom)), (h - (h * zoom))]}
        tx, ty = positions[position]
        M = np.array([[zoom, 0, tx], [0, zoom, ty]])
        frame = cv2.warpAffine(frame, M, (w, h))
        return frame
    return clip.fl(main)


def add_move_effect(clip, direction="left", move_raito=0.95):

    orig_width = clip.size[0]
    orig_height = clip.size[1]

    new_width = int(orig_width / move_raito)
    new_height = int(orig_height / move_raito)
    clip = clip.resize(width=new_width, height=new_height)

    if direction == "left":
        start_position = (0, 0)
        end_position = (orig_width - new_width, 0) 
    elif direction == "right":
        start_position = (orig_width - new_width, 0)
        end_position = (0, 0)

    duration = clip.duration
    moving_clip = clip.set_position(
        lambda t: (start_position[0] + (
            end_position[0] - start_position[0]) / duration * t, start_position[1])
    )

    final_clip = CompositeVideoClip([moving_clip], size=(orig_width, orig_height))

    return final_clip


def add_slide_effect(clips, slide_duration):
    ####### CAUTION: requires at least `slide_duration` of silence at the end of each clip #######
    durations = [clip.duration for clip in clips]
    first_clip = CompositeVideoClip(
        [clips[0].fx(transfx.slide_out, duration=slide_duration, side="left")]
    ).set_start(0)

    slide_out_sides = ["left"]
    videos = [first_clip]

    out_to_in_mapping = {"left": "right", "right": "left"}
    
    for idx, clip in enumerate(clips[1: -1], start=1):
        # For all other clips in the middle, we need them to slide in to the previous clip and out for the next one

        # determine `slide_in_side` according to the `slide_out_side` of the previous clip
        slide_in_side = out_to_in_mapping[slide_out_sides[-1]]
        
        slide_out_side = "left" if random.random() <= 0.5 else "right"
        slide_out_sides.append(slide_out_side)
                
        videos.append(
            (
                CompositeVideoClip(
                    [clip.fx(transfx.slide_in, duration=slide_duration, side=slide_in_side)]
                )
                .set_start(sum(durations[:idx]) - (slide_duration) * idx)
                .fx(transfx.slide_out, duration=slide_duration, side=slide_out_side)
            )
        )
    
    last_clip = CompositeVideoClip(
        [clips[-1].fx(transfx.slide_in, duration=slide_duration, side=out_to_in_mapping[slide_out_sides[-1]])]
    ).set_start(sum(durations[:-1]) - slide_duration * (len(clips) - 1))
    videos.append(last_clip)

    video = CompositeVideoClip(videos)
    return video


def compose_video(story_dir: Union[str, Path],
                  save_path: Union[str, Path],
                  captions: List,
                  music_path: Union[str, Path],
                  num_pages: int,
                  fps: int = 10,
                  audio_sample_rate: int = 16000,
                  audio_codec: str = "mp3",
                  caption_config: dict = {},
                  fade_duration: float = 1.0,
                  slide_duration: float = 0.4,
                  zoom_speed: float = 0.5,
                  move_ratio: float = 0.95,
                  sound_volume: float = 0.2,
                  music_volume: float = 0.2,
                  bg_speech_ratio: float = 0.4):
    if not isinstance(story_dir, Path):
        story_dir = Path(story_dir)

    sound_dir = story_dir / "sound"
    image_dir = story_dir / "image"
    speech_dir = story_dir / "speech"

    video_clips = []
    # audio_durations = []
    cur_duration = 0
    timestamps = []

    for page in trange(1, num_pages + 1):
        ##### speech track
        slide_silence = AudioArrayClip(np.zeros((int(audio_sample_rate * slide_duration), 2)), fps=audio_sample_rate)
        fade_silence = AudioArrayClip(np.zeros((int(audio_sample_rate * fade_duration), 2)), fps=audio_sample_rate)

        if (speech_dir / f"p{page}.wav").exists(): # single speech file
            single_utterance = True
            speech_file = (speech_dir / f"./p{page}.wav").__str__()
            speech_clip = AudioFileClip(speech_file, fps=audio_sample_rate)
            # speech_clip = speech_clip.audio_fadein(fade_duration)
            
            speech_clip = concatenate_audioclips([fade_silence, speech_clip, fade_silence])
        else: # multiple speech files
            single_utterance = False
            speech_files = list(speech_dir.glob(f"p{page}_*.wav"))
            speech_files = sorted(speech_files, key=lambda x: int(x.stem.split("_")[-1]))
            speech_clips = []
            for utt_idx, speech_file in enumerate(speech_files):
                speech_clip = AudioFileClip(speech_file.__str__(), fps=audio_sample_rate)
                # add multiple timestamps of the same speech clip
                if utt_idx == 0:
                    timestamps.append([cur_duration + fade_duration,
                                       cur_duration + fade_duration + speech_clip.duration])
                    cur_duration += speech_clip.duration + fade_duration
                elif utt_idx == len(speech_files) - 1:
                    timestamps.append([
                        cur_duration,
                        cur_duration + speech_clip.duration
                    ])
                    cur_duration += speech_clip.duration + fade_duration + slide_duration
                else:
                    timestamps.append([
                        cur_duration,
                        cur_duration + speech_clip.duration
                    ])
                    cur_duration += speech_clip.duration
                speech_clips.append(speech_clip)
            speech_clip = concatenate_audioclips([fade_silence] + speech_clips + [fade_silence])
            speech_file = speech_files[0] # for energy calculation
        
        # add slide silence
        if page == 1:
            speech_clip = concatenate_audioclips([speech_clip, slide_silence])
        else:
            speech_clip = concatenate_audioclips([slide_silence, speech_clip, slide_silence])
        
        # add the timestamp of the whole clip as a single element 
        if single_utterance:
            if page == 1:
                timestamps.append([cur_duration + fade_duration,
                                   cur_duration + speech_clip.duration - fade_duration - slide_duration])
                cur_duration += speech_clip.duration - slide_duration
            else:
                timestamps.append([cur_duration + fade_duration + slide_duration,
                                   cur_duration + speech_clip.duration - fade_duration - slide_duration])
                cur_duration += speech_clip.duration - slide_duration

        speech_array, _ = librosa.core.load(speech_file, sr=None)
        speech_rms = librosa.feature.rms(y=speech_array)[0].mean()

        # set image as the main content, align the duration
        image_file = (image_dir / f"./p{page}.png").__str__()        
        image_clip = ImageClip(image_file)
        image_clip = image_clip.set_duration(speech_clip.duration).set_fps(fps)
        image_clip = image_clip.crossfadein(fade_duration).crossfadeout(fade_duration)

        if random.random() <= 0.5: # zoom in or zoom out
            if random.random() <= 0.5:
                zoom_mode = "in"
            else:
                zoom_mode = "out"
            image_clip = add_zoom_effect(image_clip, zoom_speed, zoom_mode)
        else: # move left or right
            if random.random() <= 0.5:
                direction = "left"
            else:
                direction = "right"
            image_clip = add_move_effect(image_clip, direction=direction, move_raito=move_ratio)

        # sound track
        sound_file = sound_dir / f"p{page}.wav"
        if sound_file.exists():
            sound_clip = AudioFileClip(sound_file.__str__(), fps=audio_sample_rate)
            sound_clip = sound_clip.audio_fadein(fade_duration)
            if sound_clip.duration < speech_clip.duration:
                sound_clip = audio_loop(sound_clip, duration=speech_clip.duration)
            else:
                sound_clip = sound_clip.subclip(0, speech_clip.duration)
            sound_array, _ = librosa.core.load(sound_file.__str__(), sr=None)
            sound_rms = librosa.feature.rms(y=sound_array)[0].mean()
            ratio = speech_rms / sound_rms * bg_speech_ratio
            audio_clip = CompositeAudioClip([speech_clip, sound_clip.volumex(sound_volume * ratio).audio_fadeout(fade_duration)])
        else:
            audio_clip = speech_clip

        video_clip = image_clip.set_audio(audio_clip)        
        video_clips.append(video_clip)

        # audio_durations.append(audio_clip.duration)

    # final_clip = concatenate_videoclips(video_clips, method="compose")
    composite_clip = add_slide_effect(video_clips, slide_duration=slide_duration)
    composite_clip = add_bottom_black_area(composite_clip, black_area_height=caption_config["area_height"])
    del caption_config["area_height"]
    max_caption_length = caption_config["max_length"]
    del caption_config["max_length"]
    composite_clip = add_caption(
        captions,
        story_dir / "captions.srt",
        timestamps,
        composite_clip,
        max_caption_length,
        **caption_config
    )

    # add music track, align the duration
    music_clip = AudioFileClip(music_path.__str__(), fps=audio_sample_rate)
    music_array, _ = librosa.core.load(music_path.__str__(), sr=None)
    music_rms = librosa.feature.rms(y=music_array)[0].mean()
    ratio = speech_rms / music_rms * bg_speech_ratio
    if music_clip.duration < composite_clip.duration:
        music_clip = audio_loop(music_clip, duration=composite_clip.duration)
    else:
        music_clip = music_clip.subclip(0, composite_clip.duration)
    all_audio_clip = CompositeAudioClip([composite_clip.audio, music_clip.volumex(music_volume * ratio)])
    composite_clip = composite_clip.set_audio(all_audio_clip)
    
    composite_clip.write_videofile(save_path.__str__(),
                                   audio_fps=audio_sample_rate,
                                   audio_codec=audio_codec,)


@register_tool("slideshow_video_compose")
class SlideshowVideoComposeAgent:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def adjust_caption_config(self, width, height):
        area_height = int(height * 0.06)
        fontsize = int((width + height) / 2 * 0.025)
        return {
            "fontsize": fontsize,
            "area_height": area_height
        }

    def call(self, params):
        height = params["height"]
        width = params["width"]
        pages = params["pages"]
        params["caption"].update(self.adjust_caption_config(width, height))
        compose_video(
            story_dir=Path(params["story_dir"]),
            save_path=Path(params["story_dir"]) / "output.mp4",
            captions=pages,
            music_path=Path(params["story_dir"]) / "music/music.wav",
            num_pages=len(pages),
            fps=params["fps"],
            audio_sample_rate=params["audio_sample_rate"],
            audio_codec=params["audio_codec"],
            caption_config=params["caption"],
            **params["slideshow_effect"]
        )