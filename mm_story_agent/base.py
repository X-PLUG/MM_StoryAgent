from abc import ABC

register_map = {
    'qwen': 'QwenAgent',
    'qa_outline_story_writer': 'QAOutlineStoryWriter',
    'musicgen_t2m': 'MusicGenAgent',
    'story_diffusion_t2i': 'StoryDiffusionAgent',
    'cosyvoice_tts': 'CosyVoiceAgent',
    'audioldm2_t2a': 'AudioLDM2Agent',
    'slideshow_video_compose': 'SlideshowVideoComposeAgent',
    'freesound_sfx_retrieval': 'FreesoundSfxAgent',
    'freesound_music_retrieval': 'FreesoundMusicAgent',
}    


def import_from_register(key):
    value = register_map[key]
    exec(f'from . import {value}')


class ToolRegistry(dict):

    def _import_key(self, key):
        try:
            import_from_register(key)
        except Exception as e:
            print(f'import {key} failed, details: {e}')

    def __getitem__(self, key):
        if key not in self.keys():
            self._import_key(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self._import_key(key)
        return super().__contains__(key)
    

TOOL_REGISTRY = ToolRegistry()


def register_tool(name):

    def decorator(cls):
        TOOL_REGISTRY[name] = cls
        return cls
    
    return decorator


def init_tool_instance(cfg):
    return TOOL_REGISTRY[cfg["tool"]](cfg["cfg"])