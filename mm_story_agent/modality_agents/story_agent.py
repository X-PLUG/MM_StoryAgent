import json
from typing import Dict
import random

from tqdm import trange, tqdm

from ..utils.llm_output_check import parse_list
from ..base import register_tool, init_tool_instance
from ..prompts_en import question_asker_system, expert_system, \
    dlg_based_writer_system, dlg_based_writer_prompt, chapter_writer_system


def json_parse_outline(outline):
    outline = outline.strip("```json").strip("```")
    try:
        outline = json.loads(outline)
        if not isinstance(outline, dict):
            return False
        if outline.keys() != {"story_title", "story_outline"}:
            return False
        for chapter in outline["story_outline"]:
            if chapter.keys() != {"chapter_title", "chapter_summary"}:
                return False
    except json.decoder.JSONDecodeError:
        return False
    return True


@register_tool("qa_outline_story_writer")
class QAOutlineStoryWriter:

    def __init__(self,
                 cfg: Dict):
        self.cfg = cfg
        self.temperature = cfg.get("temperature", 1.0)
        self.max_conv_turns = cfg.get("max_conv_turns", 3)
        self.num_outline = cfg.get("num_outline", 4)
        self.llm_type = cfg.get("llm", "qwen")

    def generate_outline(self, params):
        # `params`: story setting like 
        # {
        #     "story_title": "xxx",
        #     "main_role": "xxx",
        #     ......
        # }
        asker = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": question_asker_system,
                "track_history": False
            }
        })
        expert = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": expert_system,
                "track_history": False
            }
        })

        dialogue = []
        for turn in trange(self.max_conv_turns):
            dialogue_history = "\n".join(dialogue)
            
            question, success = asker.call(
                f"Story setting: {params}\nDialogue history: \n{dialogue_history}\n",
                temperature=self.temperature
            )
            question = question.strip()
            if question == "Thank you for your help!":
                break
            dialogue.append(f"You: {question}")
            answer, success = expert.call(
                f"Story setting: {params}\nQuestion: \n{question}\nAnswer: ",
                temperature=self.temperature
            )
            answer = answer.strip()
            dialogue.append(f"Expert: {answer}")

        # print("\n".join(dialogue))
        writer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": dlg_based_writer_system,
                "track_history": False
            }
        })
        writer_prompt = dlg_based_writer_prompt.format(
            story_setting=params,
            dialogue_history="\n".join(dialogue),
            num_outline=self.num_outline
        )

        outline, success = writer.call(writer_prompt, success_check_fn=json_parse_outline)
        outline = json.loads(outline)
        # print(outline)
        return outline

    def generate_story_from_outline(self, outline):
        chapter_writer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": chapter_writer_system,
                "track_history": False
            }
        })
        all_pages = []
        for idx, chapter in enumerate(tqdm(outline["story_outline"])):
            chapter_detail, success = chapter_writer.call(
                json.dumps(
                    {
                        "completed_story": all_pages,
                        "current_chapter": chapter
                    },
                    ensure_ascii=False
                ),
                success_check_fn=parse_list,
                temperature=self.temperature
            )
            while success is False:
                chapter_detail, success = chapter_writer.call(
                    json.dumps(
                        {
                            "completed_story": all_pages,
                            "current_chapter": chapter
                        },
                        ensure_ascii=False
                    ),
                    seed=random.randint(0, 100000),
                    temperature=self.temperature,
                    success_check_fn=parse_list
                )
            pages = [page.strip() for page in eval(chapter_detail)]
            all_pages.extend(pages)
        # print(all_pages)
        return all_pages

    def call(self, params):
        outline = self.generate_outline(params)
        pages = self.generate_story_from_outline(outline)
        return pages
