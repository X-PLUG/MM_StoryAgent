import sys

from ..utils.import_utils import _LazyModule

_import_structure = {
    'story_agent': [
        'QAOutlineStoryWriter',
    ],
    'music_agent': [
        'MusicGenAgent'
    ],
    'sound_agent': [
        'AudioLDM2Agent'
    ],
    'speech_agent': [
        'CosyVoiceAgent'
    ],
    'image_agent': [
        'StoryDiffusionAgent'
    ],
    'llm': [
        'QwenAgent'
    ],
    "freesound_agent": [
        "FreesoundSfxAgent",
        "FreesoundMusicAgent"
    ]
}

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()['__file__'],
    _import_structure,
    module_spec=__spec__,
)