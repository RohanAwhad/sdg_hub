# Standard
from dataclasses import dataclass
import logging
import pprint
import random

# Third Party
from flask import Flask, request  # type: ignore[import-not-found]
from werkzeug import exceptions  # type: ignore[import-not-found]
import click  # type: ignore[import-not-found]
import yaml

def match_prompt(prompt: str, patterns: list[dict]) -> str:
  for p in patterns:
    if "contains" in p and all(c in prompt for c in p["contains"]):
      return p["response"]
    elif "contains" not in p and prompt:
      return p["response"]
  return ""



# mock openAI completion responses
# credit: https://github.com/openai/openai-python/issues/715#issuecomment-1809203346
# License: MIT
def create_chat_completion(content: str, model: str = "gpt-3.5") -> dict:
    response = {
        "id": "chatcmpl-2nYZXNHxx1PeK1u8xXcE1Fqr1U6Ve",
        "object": "chat.completion",
        "created": "12345678",
        "model": model,
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "text": content,
                "content": content,
                "index": 0,
                "logprobs": None,
                "finish_reason": "length",
            },
        ],
        "usage": {
            "prompt_tokens": random.randint(10, 500),
            "completion_tokens": random.randint(10, 500),
            "total_tokens": random.randint(10, 500),
        },
    }

    return response



# Globals
app = Flask(__name__)
strategies: list[dict]  # a read only list of matching strategies

# Routes
@app.route("/v1/completions", methods=["POST"])
def completions():
    data = request.get_json()
    if not data or "prompt" not in data:
        raise exceptions.BadRequest("prompt is empty or None")

    prompt = data.get("prompt")
    prompt_debug_str = prompt
    if len(prompt) > 90:
        prompt_debug_str = data["prompt"][:90] + "..."

    app.logger.debug(
        f"{request.method} {request.url} {data['model']} {prompt_debug_str}"
    )

    chat_response = match_prompt(
        prompt, strategies
    )  # handle prompt and generate correct response

    response = create_chat_completion(chat_response, model=data.get("model"))
    app.logger.debug(f"response: {pprint.pformat(response, compact=True)}")
    return response

# config
@dataclass
class Config:
    matches: list[dict]
    port: int = 11434
    debug: bool = False


@click.command()
@click.option(
    "-c",
    "--config",
    "config",
    type=click.File(mode="r", encoding="utf-8"),
    required=True,
    help="yaml config file",
)
def start_server(config):
    # get config
    yaml_data = yaml.safe_load(config)
    if not isinstance(yaml_data, dict):
        raise ValueError(f"config file {config} must be a set of key-value pairs")

    conf = Config(**yaml_data)

    # configure logger
    if conf.debug:
        app.logger.setLevel(logging.DEBUG)
        app.logger.debug("debug mode enabled")
    else:
        app.logger.setLevel(logging.INFO)

    # create match strategy object
    global strategies  # pylint: disable=global-statement
    strategies = conf.matches

    # init server
    app.run(debug=conf.debug, port=conf.port)


if __name__ == "__main__":
    start_server()  # pylint: disable=no-value-for-parameter
