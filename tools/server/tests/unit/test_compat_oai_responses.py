import time

import pytest
from openai import OpenAI
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()

def test_responses_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_output_tokens=8,
        temperature=0.8,
    )
    assert res.id.startswith("resp_")
    assert res.output[0].id is not None
    assert res.output[0].id.startswith("msg_")
    assert match_regex("(Suddenly)+", res.output_text)
    assert res.created_at is not None
    assert isinstance(res.created_at, (int, float))
    assert abs(res.created_at - time.time()) < 10
    assert res.model is not None
    assert len(res.model) > 0

def test_responses_stream_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    stream = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_output_tokens=8,
        temperature=0.8,
        stream=True,
    )

    now = int(time.time())
    gathered_text = ''
    resp_id = ''
    msg_id = ''
    output_item_added_indices = []
    for r in stream:
        if r.type == "response.created":
            assert r.response.id.startswith("resp_")
            resp_id = r.response.id
            assert r.response.created_at is not None
            assert isinstance(r.response.created_at, (int, float))
            assert abs(r.response.created_at - now) < 10
            assert r.response.model is not None
            assert len(r.response.model) > 0
        if r.type == "response.in_progress":
            assert r.response.id == resp_id
            assert r.response.created_at is not None
            assert r.response.model is not None
        if r.type == "response.output_item.added":
            assert r.item.id is not None
            assert r.item.id.startswith("msg_")
            msg_id = r.item.id
            assert r.output_index is not None
            output_item_added_indices.append(r.output_index)
        if (r.type == "response.content_part.added" or
            r.type == "response.output_text.delta" or
            r.type == "response.output_text.done" or
            r.type == "response.content_part.done"):
            assert r.item_id == msg_id
            assert r.output_index is not None
            assert r.content_index == 0
        if r.type == "response.output_item.done":
            assert r.item.id == msg_id
            assert r.output_index is not None

        if r.type == "response.output_text.delta":
            gathered_text += r.delta
        if r.type == "response.completed":
            assert r.response.id.startswith("resp_")
            assert r.response.output[0].id is not None
            assert r.response.output[0].id.startswith("msg_")
            assert gathered_text == r.response.output_text
            assert match_regex("(Suddenly)+", r.response.output_text)
    assert len(output_item_added_indices) > 0
    assert output_item_added_indices == list(range(len(output_item_added_indices))), \
        f"output_index values should be sequential, got {output_item_added_indices}"
