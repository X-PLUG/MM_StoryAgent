from typing import List
import os
import shutil
from pathlib import Path
import json
import urllib.parse
import requests

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from ..prompts_en import fsd_search_reviser_system, fsd_search_reviewer_system, fsd_music_reviser_system, fsd_music_reviewer_system
from ..base import register_tool, init_tool_instance
from ..utils.llm_output_check import parse_list


def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

    except Exception as e:
        print(f"Error during downloading: {e}")

def search_download_sound(query, save_path, max_duration=10.0):
    url = f"https://freesound.org/apiv2/search/text/"
    params = {
        "query": urllib.parse.quote(query),
        "token": os.environ["FREESOUND_API_KEY"],
        "filter": urllib.parse.quote(f"duration:[0 TO {max_duration}]")
    }
    response = requests.get(url, params=params)
    response = response.json()
    if response["count"] > 0:
        fsd_id = response["results"][0]["id"]
        id_url = f"https://freesound.org/apiv2/sounds/{fsd_id}/"
        response = requests.get(id_url, params={"token": os.environ["FREESOUND_API_KEY"]})
        sound_detail = response.json()
        download_file(sound_detail["previews"]["preview-hq-mp3"], save_path)


def search_download_mix_query_list(query_list, save_path, sample_rate: int = 16000):
    save_path = Path(save_path)
    tmp_path = save_path.parent / save_path.stem
    tmp_path.mkdir(exist_ok=True, parents=True)
    for idx, query in enumerate(query_list):
        search_download_sound(query, tmp_path / f"{idx}.mp3")
    # resample all x.mp3 to the same sample rate, single channel, and mix them to create a single file
    # using librosa
    mixed_audio = None
    for audio_file in tmp_path.glob("*.mp3"):
        y, sr = librosa.load(audio_file, sr=sample_rate, mono=True)
        if mixed_audio is None:
            mixed_audio = y
        else:
            if y.shape[0] > mixed_audio.shape[0]:
                mixed_audio = np.pad(mixed_audio, (0, y.shape[0] - mixed_audio.shape[0]))
            elif y.shape[0] < mixed_audio.shape[0]:
                y = np.pad(y, (0, mixed_audio.shape[0] - y.shape[0]))
            mixed_audio += y
    sf.write(save_path.__str__(), mixed_audio, sample_rate)
    shutil.rmtree(tmp_path)


@register_tool("freesound_sfx_retrieval")
class FreesoundSfxAgent:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def generate_search_query_from_story(
            self,
            pages: List,
        ):
        query_reviser = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": fsd_search_reviser_system,
                "track_history": False
            }
        })
        query_reviewer = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": fsd_search_reviewer_system,
                "track_history": False
            }
        })
        num_turns = self.cfg.get("num_turns", 3)

        query_lists = []
        for page in pages:
            review = ""
            query_list = ""
            for turn in range(num_turns):
                query_list, success = query_reviser.call(
                    json.dumps({
                        "story": page,
                        "previous_result": query_list,
                        "improvement_suggestions": review,
                    }, ensure_ascii=False),
                    success_check_fn=parse_list
                )
                review, success = query_reviewer.call(json.dumps({
                    "story": page,
                    "sound_description": query_list
                }, ensure_ascii=False))
                if review == "Check passed.":
                    break
                else:
                    print(review)
            query_lists.append(eval(query_list))

        return query_lists

    def call(self, params):
        queries = self.generate_search_query_from_story(params["pages"])
        save_path = params["save_path"]
        save_path = Path(save_path)
        for idx, query_list in enumerate(tqdm(queries)):
            search_download_mix_query_list(
                query_list,
                save_path / f"p{idx + 1}.mp3",
                params.get("sample_rate", 16000)
            )
        return {
            "queries": queries
        }

@register_tool("freesound_music_retrieval")
class FreesoundMusicAgent:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def generate_search_query_from_story(
            self,
            pages: List,
        ):
        query_reviser = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": fsd_music_reviser_system,
                "track_history": False
            }
        })
        query_reviewer = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": fsd_music_reviewer_system,
                "track_history": False
            }
        })
        num_turns = self.cfg.get("num_turns", 3)

        query = ""
        review = ""

        for turn in range(num_turns):
            query, success = query_reviser.call(
                json.dumps({
                    "story": pages,
                    "previous_result": query,
                    "improvement_suggestions": review,
                }, ensure_ascii=False)
            )
            review, success = query_reviewer.call(json.dumps({
                "story": pages,
                "music_query": query
            }, ensure_ascii=False))
            if review == "Check passed.":
                break
            else:
                print(review)

        return query

    def call(self, params):
        query = self.generate_search_query_from_story(params["pages"])
        save_path = params["save_path"]
        save_path = Path(save_path)
        search_download_sound(
            query,
            save_path / "tmp.mp3",
            max_duration=60.0
        )
        sample_rate = params.get("sample_rate", 16000)
        y, sr = librosa.load(save_path / "tmp.mp3", sr=sample_rate, mono=True)
        sf.write(save_path / "music.wav", y, sample_rate)
        (save_path / "tmp.mp3").unlink()
        return {
            "music_query": query
        }
