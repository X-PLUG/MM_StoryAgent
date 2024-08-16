from pathlib import Path
from typing import List, Dict
import json

import torch
import soundfile as sf
from diffusers import AudioLDM2Pipeline

from mm_story_agent.prompts_en import story_to_sound_reviser_system, story_to_sound_review_system
from mm_story_agent.base import register_tool, init_tool_instance


class AudioLDM2Synthesizer:

    def __init__(self,
                 device: str = 'cuda',
                 ) -> None:
        self.device = device
        self.pipe = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2",
            torch_dtype=torch.float16
        ).to(self.device)
    
    def call(
        self,
        prompts: List[str],
        n_candidate_per_text: int = 3,
        seed: int = 0,
        guidance_scale: float = 3.5,
        ddim_steps: int = 100,
    ):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        audios = self.pipe(
            prompts, 
            num_inference_steps=ddim_steps, 
            audio_length_in_s=10.0,
            guidance_scale=guidance_scale,
            generator=generator,
            num_waveforms_per_prompt=n_candidate_per_text).audios
        
        audios = audios[::n_candidate_per_text]

        return audios


@register_tool("audioldm2_t2a")
class AudioLDM2Agent:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def call(self, params: Dict):
        pages: List = params["pages"]
        save_path: str = params["save_path"]
        sound_prompts = self.generate_sound_prompt_from_story(pages,)
        save_paths = []
        forward_prompts = []
        save_path = Path(save_path)
        for idx in range(len(pages)):
            if sound_prompts[idx] != "No sounds.":
                save_paths.append(save_path / f"p{idx + 1}.wav")
                forward_prompts.append(sound_prompts[idx])
        
        generation_agent = AudioLDM2Synthesizer(device=self.cfg.get("device", "cuda"))
        if len(forward_prompts) > 0:
            sounds = generation_agent.call(
                forward_prompts,
                n_candidate_per_text=params.get("n_candidate_per_text", 3),
                seed=params.get("seed", 0),
                guidance_scale=params.get("guidance_scale", 3.5),
                ddim_steps=params.get("ddim_steps", 100),
            )
            for sound, path in zip(sounds, save_paths):
                sf.write(path.__str__(), sound, self.cfg["sample_rate"])
        return {
            "prompts": sound_prompts,
        }

    def generate_sound_prompt_from_story(
            self,
            pages: List,
        ):
        sound_prompt_reviser = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": story_to_sound_reviser_system,
                "track_history": False
            }
        })
        sound_prompt_reviewer = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": story_to_sound_review_system,
                "track_history": False
            }
        })
        num_turns = self.cfg.get("num_turns", 3)

        sound_prompts = []
        for page in pages:
            review = ""
            sound_prompt = ""
            for turn in range(num_turns):
                sound_prompt, success = sound_prompt_reviser.call(json.dumps({
                    "story": page,
                    "previous_result": sound_prompt,
                    "improvement_suggestions": review,
                }, ensure_ascii=False))
                if sound_prompt.startswith("Sound description:"):
                    sound_prompt = sound_prompt[len("Sound description:"):]
                review, success = sound_prompt_reviewer.call(json.dumps({
                    "story": page,
                    "sound_description": sound_prompt
                }, ensure_ascii=False))
                if review == "Check passed.":
                    break
                # else:
                    # print(review)
            sound_prompts.append(sound_prompt)

        return sound_prompts

