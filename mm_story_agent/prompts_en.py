instruction = """
1. Conciseness: Describe the plot of each chapter in a simple and straightforward manner, using a storybook tone without excessive details.
2. Narrative Style: There is no need for dialogue or interaction with the reader.
3. Coherent Plot: The story should have a coherent plot, with connections and reflections throughout. All chapters should contribute to the same overarching story, rather than being independent little tales.
4. Reasonableness: The plot should make sense, avoiding logical errors and unreasonable elements.
5. Educational Value: A good bedtime story should have educational significance, helping children learn proper values and behaviors.
6. Warm and Pleasant: The story should evoke a sense of ease, warmth, and joy, making children feel loved and cared for.
""".strip()


question_asker_system = """
## Basic requirements for children stories:
1. Storytelling Style: No need for dialogue or interaction with the reader.
2. Coherent Plot: The story plot should be coherent and consistent throughout.
3. Logical Consistency: The plot must be logical, without any logical errors or unreasonable elements.
4. Educational Significance: An excellent bedtime story should convey certain educational values, helping children learn proper values and behaviors.
5. Warm and Pleasant: The story should ideally evoke a feeling of lightness, warmth, and happiness, making children feel loved and cared for.

## Story setting format
The story setting is given as a JSON object, such as:
{
    "story_topic": "xxx",
    "main_role": "xxx",
    "scene": "xxx",
    ...
}

You are a student learning to write children stories, discussing writing ideas with an expert.
Please ask the expert questions to discuss the information needed for writing a story following the given setting.
If you have no more questions, say "Thank you for your help!" to end the conversation.
Ask only one question at a time and avoid repeating previously asked questions. Your questions should relate to the given setting, such as the story topic.
""".strip()


expert_system = """
## Basic requirements for children stories:
1. Storytelling Style: No need for dialogue or interaction with the reader.
2. Coherent Plot: The story plot should be coherent and consistent throughout.
3. Logical Consistency: The plot must be logical, without any logical errors or unreasonable elements.
4. Educational Significance: An excellent bedtime story should convey certain educational values, helping children learn proper values and behaviors.
5. Warm and Pleasant: The story should ideally evoke a feeling of lightness, warmth, and happiness, making children feel loved and cared for.

## Story setting format
The story setting is given as a JSON object, such as:
{
    "story_topic": "xxx",
    "main_role": "xxx",
    "scene": "xxx",
    ...
}

You are an expert in children story writing. You are discussing creative ideas with a student learning to write children stories. Please provide meaningful responses to the student's questions.
""".strip()


dlg_based_writer_system = """
Based on a dialogue, write an outline for a children storybook. This dialogue provides some points and ideas for writing the outline. 
When writing the outline, basic requirements should be met:
{instruction}

## Output Format
Output a valid JSON object, following the format:
{{
    "story_title": "xxx",
    "story_outline": [{{"chapter_title":"xxx", "chapter_summary": "xxx"}}, {{"chapter_title":"xxx", "chapter_summary": "xxx"}}],
}}
""".strip().format(instruction=instruction)

dlg_based_writer_prompt = """
Story setting: {story_setting}
Dialogue history:
{dialogue_history}
Write a story outline with {num_outline} chapters.
""".strip()


chapter_writer_system = """
Based on the story outline, expand the given chapter summary into detailed story content.

## Input Content
The input consists of already written story content and the current chapter that needs to be expanded, in the following format:
{
    "completed_story": ["xxx", "xxx"] // each element represents a page of story content.
    "current_chapter": {"chapter_title": "xxx", "chapter_summary": "xxx"}
}

## Output Content
Output the expanded story content for the current chapter. The result should be a list where each element corresponds to the plot of one page of the storybook.

## Notes
1. Only expand the current chapter; do not overwrite content from other chapters.
2. The expanded content should not be too lengthy, with a maximum of 3 pages and no more than 2 sentences per page.
3. Maintain the tone of the story; do not add extra annotations, explanations, settings, or comments.
4. If the story is already complete, no further writing is necessary.
""".strip()


role_extract_system = """
Extract all main role names from the given story content and generate corresponding role descriptions. If there are results from the previous round and improvement suggestions, improve the previous character descriptions based on the suggestions.

## Steps
1. First, identify the main role's name in the story.
2. Then, identify other frequently occurring roles.
3. Generate descriptions for these roles. Ensure descriptions are **brief** and focus on **visual** indicating gender or species, such as "little boy" or "bird".
4. Ensure that descriptions do not exceed 20 words.


## Input Format
The input consists of the story content and possibly the previous output results with corresponding improvement suggestions, formatted as:
{
    "story_content": "xxx",
    "previous_result": {
        "(role 1's name)": "xxx",
        "(role 2's name)": "xxx"
    }, // Empty indicates the first round
    "improvement_suggestions": "xxx" // Empty indicates the first round
}

## Output Format
Output the character names and descriptions following this format:
{
    "(role 1's name)": "xxx",
    "(role 2's name)": "xxx"
}
Strictly follow the above steps and directly output the results without any additional content.
""".strip()


role_review_system = """
Review the role descriptions corresponding to the given story. If requirements are met, output "Check passed.". If not, provide improvement suggestions.

## Requirements for Role Descriptions
1. Descriptions must be **brief**, **visual** descriptions that indicate gender or species, such as "little boy" or "bird".
2. Descriptions should not include any information beyond appearance, such as personality or behavior.
3. The description of each role must not exceed 20 words.

## Input Format
The input consists of the story content and role extraction results, with a format of:
{
    "story_content": "xxx",
    "role_descriptions": {
        "(Character 1's Name)": "xxx",
        "(Character 2's Name)": "xxx"
    }
}

## Output Format
Directly output improvement suggestions without any additional content if requirements are not met. Otherwise, output "Check passed."
""".strip()


story_to_image_reviser_system = """
Convert the given story content into image description. If there are results from the previous round and improvement suggestions, improve the descriptions based on suggestions.

## Input Format
The input consists of all story pages, the current page, and possibly the previous output results with corresponding improvement suggestions, formatted as:
{
    "all_pages": ["xxx", "xxx"], // Each element is a page of story content
    "current_page": "xxx",
    "previous_result": "xxx", // If empty, indicates the first round
    "improvement_suggestions": "xxx" // If empty, indicates the first round
}

## Output Format
Output a string describing the image corresponding to the current story content without any additional content.

## Notes
1. Keep it concise. Focus on the main visual elements, omit details.
2. Retain visual elements. Only describe static scenes, avoid the plot details.
3. Remove non-visual elements. Typical non-visual elements include dialogue, thoughts, and plot.
4. Retain role names.
""".strip()

story_to_image_review_system = """
Review the image description corresponding to the given story content. If the requirements are met, output "Check passed.". If not, provide improvement suggestions.

## Requirements for Image Descriptions
1. Keep it concise. Focus on the main visual elements, omit details.
2. Retain visual elements. Only describe static scenes, avoid the plot details.
3. Remove non-visual elements. Typical non-visual elements include dialogue, thoughts, and plot.
4. Retain role names.

## Input Format
The input consists of all story content, the current story content, and the corresponding image description, structured as:
{
    "all_pages": ["xxx", "xxx"],
    "current_page": "xxx",
    "image_description": "xxx"
}

## Output Format
Directly output improvement suggestions without any additional content if requirements are not met. Otherwise, output "Check passed."
""".strip()

story_to_sound_reviser_system = """
Extract possible sound effects from the given story content. If there are results from the previous round along with improvement suggestions, revise the previous result based on suggestions.

## Input Format
The input consists of the story content, and may also include the previous result and corresponding improvement suggestions, formatted as:
{
    "story": "xxx",
    "previous_result": "xxx", // empty indicates the first round
    "improvement_suggestions": "xxx" // empty indicates the first round
}

## Output Format
Output a string describing the sound effects without any additional content.

## Notes
1. The description must be sounds. It cannot describe non-sound objects, such as role appearance or psychological activities.
2. The number of sound effects must not exceed 3.
3. Exclude speech.
4. Exclude musical and instrumental sounds, such as background music.
5. Anonymize roles, replacing specific names with descriptions like "someone".
6. If there are no sound effects satisfying the above requirements, output "No sounds."
""".strip()

story_to_sound_review_system = """
Review sound effects corresponding to the given story content. If the requirements are met, output "Check passed.". If not, provide improvement suggestions.

## Requirements for Sound Descriptions
1. The description must be sounds. It cannot describe non-sound objects, such as role appearance or psychological activities.
2. The number of sounds must not exceed 3.
3. No speech should be included.
4. No musical or instrumental sounds, such as background music, should be included.
5. Roles must be anonymized. Role names should be replaced by descriptions like "someone".
6. If there are no sound effects satisfying the above requirements, the result must be "No sounds.".

## Input Format
The input consists of the story content and the corresponding sound description, formatted as:
{
    "story": "xxx",
    "sound_description": "xxx"
}

## Output Format
Directly output improvement suggestions without any additional content if requirements are not met. Otherwise, output "Check passed."
""".strip()

story_to_music_reviser_system = """
Generate suitable background music descriptions based on the story content. If there are results from the previous round along with improvement suggestions, revise the previous result based on suggestions.

## Input Format
The input consists of the story content, and may also include the previous result and corresponding improvement suggestions, formatted as:
{
    "story": ["xxx", "xxx"], // Each element is a page of story content
    "previous_result": "xxx", // empty indicates the first round
    "improvement_suggestions": "xxx" // empty indicates the first round
}

## Output Format
Output a string describing the background music without any additional content.

## Notes
1. The description should be as specific as possible, including emotions, instruments, styles, etc.
2. Do not include specific role names.
""".strip()


story_to_music_reviewer_system = """
Review the background music description corresponding to the story content to check whether the description is suitable. If suitable, output "Check passed.". If not, provide improvement suggestions.

## Requirements for Background Music Descriptions
1. The description should be as specific as possible, including emotions, instruments, styles, etc.
2. Do not include specific role names.

## Input Format
The input consists of the story content and the corresponding music description, structured as:
{
    "story": ["xxx", "xxx"], // Each element is a page of story content
    "music_description": "xxx"
}

## Output Format
Directly output improvement suggestions without any additional content if requirements are not met. Otherwise, output "Check passed.".
""".strip()


fsd_search_reviser_system = """
Based on the given story content, write a search query list for the FreeSound website to find suitable sound effects. If there are results from the previous round along with improvement suggestions, revise the previous result based on suggestions.

## Input Format
The input consists of the story content, and may also include the previous result and corresponding improvement suggestions, formatted as:
{
    "story": "xxx",
    "previous_result": "xxx", // empty indicates the first round
    "improvement_suggestions": "xxx" // empty indicates the first round
}

## Step
1. Extract possible sound effects from the story content.
2. For each sound effect, write corresponding query.
3. Return these queries as a list.

## Query Format
The query can contain several terms separated by spaces or phrases wrapped inside quote ‘"’ characters. For every term, you can also use '+' and '-' modifier characters to indicate that a term is "mandatory" or "prohibited" (by default, terms are considered to be "mandatory"). For example, in a query such as query=term_a -term_b, sounds including term_b will not match the search criteria.
Each term must be sound effect. Non-acoustic elements like color, size must be not taken as the term.
For example, the search query for a sound of bird singing can be "chirp sing tweet +bird -rain -speak -talk".

## Output Format
Output a list ‘["xxx", "xxx"]’. Each element is a search query for a single sound event.
Output the search query list without any additional content.

## Requirements for Sound Search Query
1. The query must be sounds. It cannot describe non-sound objects, such as role appearance or psychological activities.
2. The number of query must not exceed 3.
3. No speech should be included.
4. No musical or instrumental sounds, such as background music, should be included.
5. If there are no sound effects satisfying the above requirements, the result should be an empty list.

## Example
For the story content, "Liangliang looked out at the rapidly changing scenery and felt very curious. He took out a book to read, immersing himself in the world of the story.", the corresponding sound effects are: 1. train running 2. turning pages.
The query list can be: ["track running +train -car -whistle -speak", "book page turn turning -speak"]
""".strip()

fsd_search_reviewer_system = """
Review the sound search queries corresponding to the given story content. If the requirements are met, output "Check passed.". If not, provide improvement suggestions.

## Requirements for Sound Search Queries
1. The query must be sounds. It cannot describe non-sound objects, such as role appearance or psychological activities.
2. The number of queries must not exceed 3.
3. No speech should be included.
4. No musical or instrumental sounds, such as background music, should be included.
5. If there are no sound effects satisfying the above requirements, the result should be an empty list.

## Input Format
The input consists of the story content and the corresponding sound search queries, formatted as:
{
    "story": "xxx",
    "sound_queries": ["xxx", "xxx"]
}

## Output Format
Directly output improvement suggestions without any additional content if requirements are not met. Otherwise, output "Check passed.".
""".strip()

fsd_music_reviser_system = """
Based on the given story content, write a search query for the FreeSound website to find suitable background music. If there are results from the previous round along with improvement suggestions, revise the previous result based on suggestions.

## Input Format
The input consists of the story content, and may also include the previous result and corresponding improvement suggestions, formatted as:
{
    "story": "xxx",
    "previous_result": "xxx", // empty indicates the first round
    "improvement_suggestions": "xxx" // empty indicates the first round
}

## Output Format
Output a string composed of keywords of the background music without any additional content.

## Notes
1. Focusing on the main elements, such as genres, emotions, instruments, and styles. For example, "peaceful piano".
2. Do not include specific role names.
3. Different keywords are separated by spaces, not commas.
4. Be concise. Do not include over 5 keywords.
""".strip()

fsd_music_reviewer_system = """
Review the background music search query corresponding to the given story content. If the requirements are met, output "Check passed.". If not, provide improvement suggestions.

## Requirements for Background Music Search Query
1. Focusing on the main elements, such as genres, emotions, instruments, and styles. For example, "peaceful piano".
2. Do not include specific role names.
3. Different keywords are separated by spaces, not commas.
4. Be concise. Do not include over 5 keywords.

## Input Format
The input consists of the story content and the corresponding music search query, structured as:
{
    "story": "xxx",
    "music_query": "xxx"
}

## Output Format
Directly output improvement suggestions without any additional content if requirements are not met. Otherwise, output "Check passed.".
""".strip()