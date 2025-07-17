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

# Matching
# Standard
from abc import abstractmethod
from typing import Protocol
import pprint


class Match(Protocol):
    """
    Match represents a single prompt matching
    strategy. When a match is successful,
    the response is what should be returned.
    """

    response: str

    @abstractmethod
    def match(self, prompt: str) -> str | None:
        raise NotImplementedError


class Always:
    """
    Always is a matching strategy that always
    is a positive match on a given prompt.

    This is best used when only one prompt response
    is expected.
    """

    def __init__(self, response: str):
        self.response = response

    def match(self, prompt: str) -> str | None:
        if prompt:
            return self.response
        return None


class Contains:
    """
    Contains is a matching strategy that checks
    if the prompt string contains all of
    the substrings in the `contains` attribute.
    """

    contains: list[str]

    def __init__(self, contains: list[str], response: str):
        if not contains or len(contains) == 0:
            raise ValueError("contains must not be empty")
        self.response = response
        self.contains = contains

    def match(self, prompt: str) -> str | None:
        if not prompt:
            return None
        for context in self.contains:
            if context not in prompt:
                return None

        return self.response


# helper function pulled out for easier testing
def to_match(pattern: dict) -> Match:
    response = pattern.get("response")
    if not response:
        raise ValueError(
            f"matching strategy must have a response: {pprint.pformat(pattern, compact=True)}"
        )
    if "contains" in pattern:
        return Contains(**pattern)
    return Always(**pattern)


class Matcher:
    """
    Matcher matches prompt context and then
    selects a user provided reply.
    """

    strategies: list[Match]

    def __init__(self, matching_patterns: list[dict]):
        if not matching_patterns:
            raise ValueError(
                "matching strategies must contain at least one Match strategy"
            )

        self.strategies: list[Match] = []
        for matching_pattern in matching_patterns:
            self.strategies.append(to_match(matching_pattern))

    def find_match(self, prompt: str) -> str:
        for strategy in self.strategies:
            response = strategy.match(prompt)
            if response:
                return response
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
strategies: Matcher  # a read only list of matching strategies

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

    chat_response = strategies.find_match(
        prompt
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
    strategies = Matcher(conf.matches)

    # init server
    app.run(debug=conf.debug, port=conf.port)


if __name__ == "__main__":
    start_server()  # pylint: disable=no-value-for-parameter
