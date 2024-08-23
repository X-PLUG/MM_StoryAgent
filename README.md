# MM-StoryAgent
This repo is the official implementation of "MM-StoryAgent: Immersive Narrated Storybook Video Generation with a Multi-Agent Paradigm across Text, Image and Audio".

## Introduction
MM-StoryAgent is a multi-agent framework that employs LLMs and diverse expert tools across several modalities to produce expressive storytelling videos. It hightlights in the following aspects:
* MM-StoryAgent designs a reliable and **customizable** workflow. Users can define their own expert tools to improve the generation quality of each component.
* MM-StoryAgent writes **high-quality** stories based on the input story setting, in a multi-agent, multi-stage pipeline.
* Agents of all modalities (image, speech, sound, music) generated corresponding assets are composed to an **immersive** storytelling video.

<div align="center">
    <img src="./assets/framework.png" alt="Framework" style="width: 80%;">
</div>


Besides, we provide a story topic list and story evaluation criteria for further story writing evaluation.

## News
* Aug 16, 2024: The initial version of MM-StoryAgent was released.

## Demo Video
The demo video is available:

<div align="center">
    <a href="https://www.youtube.com/watch?v=2HXGrA8mg90" target="_blank">
        <img src="https://res.cloudinary.com/marcomontalbano/image/upload/v1723627863/video_to_markdown/images/youtube--2HXGrA8mg90-c05b58ac6eb4c4700831b2b3070cd403.jpg" alt="MM-StoryAgent demo" style="width: 60%;"/>
    </a>
</div>



## Installation
Install the required dependencies and install this repo as a package:
```bash
pip install -r requirements.txt
pip install -e .
```

## Quickstart
MM-StoryAgent can be called by configuration files:
```bash
python run.py -c configs/mm_story_agent.yaml
```
Each agent is called in the following format:
```yaml
story_writer: # agent name
    tool: qa_outline_story_writer # name registered in the definition
    cfg: # parameters for initializing the agent instance
        max_conv_turns: 3
        ...
    params: # parameters for calling the agent instance
        story_topic: "Time Management: A child learning how to manage their time effectively."
        ...
```
The customization of new agents can refer to [music_agent.py](mm_story_agent/modality_agents/music_agent.py#L42). The agent class should implement `__init__` and `call` to work properly, like the following:
```python
from typing import Dict
from mm_story_agent.base import register_tool

@register_tool("my_speech_agent")
class MySpeechAgent:
    
    def __init__(self, cfg: Dict):
        # For example, the agent need `attr1` and `attr2` for initilization
        self.attr1 = cfg.attr1
        self.attr2 = cfg.attr2
        ...
    
    def call(self, params: Dict):
        # For example, calling the agent needs `voice` and `speed` parameters
        voice = params["voice"]
        speed = params["speed"]
        ...
    
```
Then the agent can be called by simply modifying the configuration like:
```yaml
speech_generation:
    tool: my_speech_agent
    cfg:
        attr1: val1
        attr2: val2
    params:
        voice: en_female
        speed: 1.0
```

## Evaluation Data
The evaluation topics are provided in [story_topics.json](story_eval/story_topics.json). Evaluation rubrics and prompts are also provided accordingly.

### Story Content Evaluation
We use GPT-4 to automatically evaluate the story quality according to several aspects.
Our story writing agent is compared with directly prompting LLM to write stories.
Evaluation scores show the advantage of our multi-agent, multi-stage story writing pipeline.

| Rubric Grading            |              | Attractiveness | Warmth | Education | Average |
|---------------------------|--------------|----------------|--------|-----------|---------|
| **Topic 1: Self-growing** | Direct       | 3.68           | 4.42   | 4.84      | 4.31    |
|                           | Story Agent  | 4.1            | 4.5    | 4.80      | **4.47**|
| **Topic 2: Family & Friendship** | Direct   | 3.94           | 5.0    | 4.72      | 4.55    |
|                           | Story Agent  | 4.36           | 4.8    | 4.92      | **4.69**|
| **Topic 3: Environments** | Direct       | 4.0            | 4.62   | 4.92      | 4.51    |
|                           | Story Agent  | 4.44           | 4.68   | 4.86      | **4.66**|
| **Topic 4: Knowledge Learning** | Direct | 4.46           | 4.14   | 4.86      | 4.49    |
|                           | Story Agent  | 4.84           | 4.52   | 4.90      | **4.75**|
| **All**                   | Direct       | 4.02           | 4.55   | 4.84      | 4.47    |
|                           | Story Agent  | 4.44           | 4.63   | 4.87      | **4.65**|



## Citation