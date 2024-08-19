# MM-StoryAgent
This repo is the official implementation of MM-StoryAgent: Immersive Narrated Storybook Video Generation with a Multi-Agent Paradigm across Text, Image and Audio.

## Demo Video
The demo video is available:

<div align="center">
    <a href="https://www.youtube.com/watch?v=2HXGrA8mg90" target="_blank">
        <img src="https://res.cloudinary.com/marcomontalbano/image/upload/v1723627863/video_to_markdown/images/youtube--2HXGrA8mg90-c05b58ac6eb4c4700831b2b3070cd403.jpg" alt="MM-StoryAgent demo" style="width: 60%;"/>
    </a>
</div>



## Usage
Install the required dependencies and install this repo as a package:
```bash
pip install -r requirements.txt
pip install -e .
```
Then MM-StoryAgent can be called by configuration files:
```bash
python run.py -c configs/mm_story_agent.yaml
```
Each agent can be called in this format:
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
The customization of new agents can refer to [music_agent.py](mm_story_agent/modality_agents/music_agent.py#L42). The agent class should implement `__init__` and `call` to work properly.
Then corresponding agent names and parameters should also be set properly.

## Evaluation Data
The evaluation topics are provided in [story_topics.json](story_eval/story_topics.json). Evaluation rubrics and prompts are also provided accordingly.

## Citation