import argparse
import yaml
from mm_story_agent import MMStoryAgent


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)

    args = parser.parse_args()

    with open(args.config, "r") as reader:
        config = yaml.load(reader, Loader=yaml.FullLoader)
    
    mm_story_agent = MMStoryAgent()
    mm_story_agent.call(config)
