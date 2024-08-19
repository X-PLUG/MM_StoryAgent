# borrrowed from STORM (https://github.com/stanford-oval/storm/blob/main/eval/eval_prometheus_no_ref.prompt).

eval_prompt_template = """
###Story Topic to Evaluate:
{story_topic}

###Children Story to Evaluate:
{story}

###Task Description:
Evaluate the children story written based on the "story topic" according to the "scoring criteria". 
1. Write a detailed feedback based on the given scoring criteria. Strictly evaluate the quality of the story according to the criteria, rather than giving general comments.
2. After writing a feedback, give a score, which is an integer between 1 and 5. You should refer to the scoring criteria for scoring.
3. The output format should be as follows: "Feedback: (evaluation based on the criteria) Score: (an integer between 1 and 5)".
4. Do not generate any other opening, closing, or explanations.

###Scoring Criteria:
[{criteria_description}]
Score 1: {score1_description}
Score 2: {score2_description}
Score 3: {score3_description}
Score 4: {score4_description}
Score 5: {score5_description}

###Feedback:
""".strip()