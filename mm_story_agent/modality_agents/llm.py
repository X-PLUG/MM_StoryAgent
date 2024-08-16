from typing import Dict, Callable
import os

from dashscope import Generation

from mm_story_agent.base import register_tool


@register_tool("qwen")
class QwenAgent(object):

    def __init__(self,
                 config: Dict):
        
        self.system_prompt = config.get("system_prompt")
        track_history = config.get("track_history", False)
        if self.system_prompt is None:
            self.history = []
        else:
            self.history = [
                {"role": "system", "content": self.system_prompt}
            ]
        self.track_history = track_history
    
    def basic_success_check(self, response):
        if not response or not response.output or not response.output.text:
            print(response)
            return False
        else:
            return True
    
    def call(self,
             prompt: str,
             model_name: str = "qwen2-72b-instruct",
             top_p: float = 0.95,
             temperature: float = 1.0,
             seed: int = 1,
             max_length: int = 1024,
             max_try: int = 5,
             success_check_fn: Callable = None
             ):
        self.history.append({
            "role": "user",
            "content": prompt
        })
        success = False
        try_times = 0
        while try_times < max_try:
            response = Generation.call(
                model=model_name,
                messages=self.history,
                top_p=top_p,
                temperature=temperature,
                api_key=os.environ.get('DASHSCOPE_API_KEY'),
                seed=seed,
                max_length=max_length
            )
            if success_check_fn is None:
                success_check_fn = lambda x: True
            if self.basic_success_check(response) and success_check_fn(response.output.text):
                response = response.output.text
                self.history.append({
                    "role": "assistant",
                    "content": response
                })
                success = True
                break
            else:
                try_times += 1
        
        if not self.track_history:
            if self.system_prompt is not None:
                self.history = self.history[:1]
            else:
                self.history = []
        
        return response, success
   