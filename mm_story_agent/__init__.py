import sys

from .utils.import_utils import _LazyModule


_import_structure = {
    'modality_agents': [
        'QAOutlineStoryWriter',
        'MusicGenAgent',
        'AudioLDM2Agent',
        'CosyVoiceAgent',
        'StoryDiffusionAgent',
        'QwenAgent',
        'FreesoundSfxAgent',
        'FreesoundMusicAgent'
    ],
    'mm_story_agent': [
        'MMStoryAgent'
    ],
    'video_compose_agent': [
        'SlideshowVideoComposeAgent'
    ],
    'base': [
        'init_tool_instance'
    ],
}

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()['__file__'],
    _import_structure,
    module_spec=__spec__,
)