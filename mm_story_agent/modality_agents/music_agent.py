from pathlib import Path
import json
from typing import List, Union, Dict

import soundfile as sf
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from mm_story_agent.prompts_en import story_to_music_reviser_system, story_to_music_reviewer_system
from mm_story_agent.base import register_tool, init_tool_instance


class MusicGenSynthesizer:

    def __init__(self,
                 model_name: str = 'facebook/musicgen-medium',
                 device: str = 'cuda',
                 sample_rate: int = 16000,
                 ) -> None:
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_name).to(device)
        self.sample_rate = sample_rate
    
    def call(self,
             prompt: Union[str, List[str]],
             save_path: Union[str, Path],
             duration: float = 30.0,
             ):
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        seq_length = int(51.2 * duration)
        wav = self.model.generate(**inputs, max_new_tokens=seq_length)[0, 0].cpu()
        wav = torchaudio.functional.resample(wav, self.model.config.audio_encoder.sampling_rate, self.sample_rate)
        sf.write(save_path, wav.numpy(), self.sample_rate)


@register_tool("musicgen_t2m")
class MusicGenAgent:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def generate_music_prompt_from_story(
            self,
            pages: List,
        ):
        music_prompt_reviser = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": story_to_music_reviser_system,
                "track_history": False
            }
        })
        music_prompt_reviewer = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": story_to_music_reviewer_system,
                "track_history": False
            }
        })

        music_prompt = ""
        review = ""
        for turn in range(self.cfg.get("max_turns", 3)):
            music_prompt, success = music_prompt_reviser.call(json.dumps({
                "story": pages,
                "previous_result": music_prompt,
                "improvement_suggestions": review,
            }, ensure_ascii=False))
            review, success = music_prompt_reviewer.call(json.dumps({
                "story_content": pages,
                "music_description": music_prompt
            }, ensure_ascii=False))
            if review == "Check passed.":
                break
        
        return music_prompt

    def call(self, params: Dict):
        pages: List = params["pages"]
        save_path: str = params["save_path"]
        save_path = Path(save_path)
        music_prompt = self.generate_music_prompt_from_story(pages)
        generation_agent = MusicGenSynthesizer(
            model_name=self.cfg.get("model_name", "facebook/musicgen-medium"),
            device=self.cfg.get("device", "cuda"),
            sample_rate=self.cfg.get("sample_rate", 16000),
        )
        generation_agent.call(
            prompt=music_prompt,
            save_path=save_path / "music.wav",
            duration=params.get("duration", 30.0),
        )
        return {
            "prompt": music_prompt,
        }