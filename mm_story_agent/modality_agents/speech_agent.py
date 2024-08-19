import os
import json
from pathlib import Path
from typing import List, Dict

from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
import nls

from mm_story_agent.base import register_tool


# Due to the trouble regarding environment, we use dashscope to deploy and call the API for CosyVoice.
class CosyVoiceSynthesizer:

    def __init__(self) -> None:
        self.access_key_id = os.environ.get('ALIYUN_ACCESS_KEY_ID')
        self.access_key_secret = os.environ.get('ALIYUN_ACCESS_KEY_SECRET')
        self.app_key = os.environ.get('ALIYUN_APP_KEY')
        self.setup_token()

    def setup_token(self):
        client = AcsClient(self.access_key_id, self.access_key_secret,
                           'cn-shanghai')
        request = CommonRequest()
        request.set_method('POST')
        request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
        request.set_version('2019-02-28')
        request.set_action_name('CreateToken')

        try:
            response = client.do_action_with_exception(request)
            jss = json.loads(response)
            if 'Token' in jss and 'Id' in jss['Token']:
                token = jss['Token']['Id']
                self.token = token
        except Exception as e:
            import traceback
            raise RuntimeError(
                f'Request token failed with error: {e}, with detail {traceback.format_exc()}'
            )

    def call(self, save_file, transcript, voice="longyuan", sample_rate=16000):
        writer = open(save_file, "wb")
        return_data = b''

        def write_data(data, *args):
            nonlocal return_data
            return_data += data
            if writer is not None:
                writer.write(data)

        def raise_error(error, *args):
            raise RuntimeError(
                f'Synthesizing speech failed with error: {error}')

        def close_file(*args):
            if writer is not None:
                writer.close()

        sdk = nls.NlsStreamInputTtsSynthesizer(
            url='wss://nls-gateway-cn-beijing.aliyuncs.com/ws/v1',
            token=self.token,
            appkey=self.app_key,
            on_data=write_data,
            on_error=raise_error,
            on_close=close_file,
        )

        sdk.startStreamInputTts(voice=voice, sample_rate=sample_rate, aformat='wav')
        sdk.sendStreamInputTts(transcript,)
        sdk.stopStreamInputTts()


@register_tool("cosyvoice_tts")
class CosyVoiceAgent:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def call(self, params: Dict):
        pages: List = params["pages"]
        save_path: str = params["save_path"]
        generation_agent = CosyVoiceSynthesizer()

        for idx, page in enumerate(pages):
            generation_agent.call(
                save_file=save_path / f"p{idx + 1}.wav",
                transcript=page,
                voice=params.get("voice", "longyuan"),
                sample_rate=self.cfg.get("sample_rate", 16000)
            )

        return {
            "modality": "speech"
        }