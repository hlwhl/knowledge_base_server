from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List, Union
import sys
import time
from flask import Flask, Response, request
from flask_cors import CORS
from llama_index import Document, GPTSimpleVectorIndex, LLMPredictor, QueryMode, QuestionAnswerPrompt, ServiceContext, PromptHelper
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import CallbackManager, BaseCallbackHandler
from langchain.schema import LLMResult
from typing import Any, Union
from queue import Queue
import threading
from pathlib import Path
import os

# enable logging of llama_index
import logging
logging.getLogger().setLevel(logging.DEBUG)


"""Callback Handler streams to stdout on new llm token."""


class CustomCallBackHandler(BaseCallbackHandler):
    queue: Queue

    def __init__(self, queue: Queue):
        self.queue = queue

    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.queue.put(token)
        # print(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.queue.put("[END]")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        self.queue.put("[END]")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        self.queue.put("[END]")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""
        self.queue.put("[END]")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""


app = Flask(__name__)
CORS(app)


@app.route('/', methods=["POST"])
def home():
    return 'Hello, World!'


openai_key = "YOUR_KEY_HERE"

print("key")
print(openai_key)

chatModel = ChatOpenAI(
    streaming=True,
    openai_api_key=openai_key,
    temperature=0)

llm_predictor = LLMPredictor(llm=chatModel)

# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 512
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(
    max_input_size, num_output, max_chunk_overlap)
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, prompt_helper=prompt_helper, chunk_size_limit=2000)


def gen_index(document, sc):
    # Document array according every 100 characters
    documents = []
    for i in range(0, len(document), 1000):
        documents.append(Document(document[i:i+1000]))

    index = GPTSimpleVectorIndex([], service_context=sc)
    for doc in documents:
        # adding space every 100 characters
        doc.text = '\n'.join([doc.text[i:i+10]
                              for i in range(0, len(doc.text), 10)])
        index.insert(doc)

    # doc = Document(document)

    # index = GPTSimpleVectorIndex.from_documents(
    #     [doc], service_context=service_context)

    return index


def getQAPrompt2():
    QUESTION_ANSWER_PROMPT_TMPL = (
        "Context information is below. This is a meeting transcript.\n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "According to context above to answer following questions\n"
        "{query_str}\n")
    QUESTION_ANSWER_PROMPT = QuestionAnswerPrompt(QUESTION_ANSWER_PROMPT_TMPL)
    return QUESTION_ANSWER_PROMPT


def getQAPrompt():
    QUESTION_ANSWER_PROMPT_TMPL = (
        "The text below is a meeting transcript, {query_str}\n\n"
        "Text: \"\"\""
        "{context_str}"
        "\n"
        "\"\"\""
    )
    QUESTION_ANSWER_PROMPT = QuestionAnswerPrompt(QUESTION_ANSWER_PROMPT_TMPL)
    return QUESTION_ANSWER_PROMPT


@ app.route("/query", methods=["POST"])
def query():
    context = request.json.get('context')
    prompt = request.json.get('prompt')

    start_time = time.time()
    print("index start @ {}".format(start_time))

    index = gen_index(context, service_context)

    end_time = time.time()
    print("index done, Total time elapsed: {}".format(end_time-start_time))

    start_time = time.time()
    print("query start @ {}".format(start_time))

    response = index.query(prompt + " respond in Chinese.",
                           text_qa_template=getQAPrompt(), response_mode="tree_summarize", mode=QueryMode.EMBEDDING, service_context=service_context, use_async=True)

    end_time = time.time()
    print("index done, Total time elapsed: {}".format(end_time-start_time))

    print(response)

    return str(response), 200


@ app.route("/stream", methods=["POST"])
def stream():
    q = Queue()
    context = request.json.get('context')
    prompt = request.json.get('prompt')
    cm = ChatOpenAI(
        streaming=True,
        openai_api_key=openai_key,
        temperature=0,
        callback_manager=CallbackManager([CustomCallBackHandler(queue=q)]), verbose=True)
    lp = LLMPredictor(llm=cm)

    sc = ServiceContext.from_defaults(
        llm_predictor=lp, prompt_helper=prompt_helper)
    index = gen_index(context, sc)

    def query():
        resp = index.query(prompt + " respond in Chinese.",
                           text_qa_template=getQAPrompt(), response_mode="tree_summarize", mode=QueryMode.EMBEDDING, service_context=sc, use_async=True)
        q.put("[END]")
        print(resp)

    t = threading.Thread(target=query)
    t.start()

    def generate():
        while True:
            token = q.get()
            if token == "[END]":
                print("stream end!")
                break
            yield token

    return Response(generate(), mimetype='text/event-stream')


mutex = threading.Lock()


@ app.route("/index/add", methods=["POST"])
def add_index():
    id = request.json.get('id')
    context = request.json.get('context')

    # adding space every 100 characters for context
    context = '\n'.join([context[i:i+10]
                         for i in range(0, len(context), 10)])

    app.logger.info(id)
    app.logger.info(context)

    index_file = os.path.join(Path('./data'), Path(id))

    if os.path.exists(index_file):
        mutex.acquire(timeout=10)
        index = GPTSimpleVectorIndex.load_from_disk(
            index_file, service_context=service_context)
        mutex.release()
        index.insert(Document(context))
        mutex.acquire(timeout=10)
        index.save_to_disk(index_file)
        mutex.release()
    else:
        mutex.acquire(timeout=10)
        index = GPTSimpleVectorIndex.from_documents(
            [Document(context)], service_context=service_context)
        index.save_to_disk(index_file)
        mutex.release()
    return "success", 200


@ app.route("/index/query", methods=["POST"])
def query_index():
    id = request.json.get('id')
    prompt = request.json.get('prompt')

    app.logger.info(id)
    app.logger.info(prompt)

    index_file = os.path.join(Path('./data'), Path(id))
    if os.path.exists(index_file):
        mutex.acquire(timeout=10)
        index = GPTSimpleVectorIndex.load_from_disk(
            index_file, service_context=service_context)
        mutex.release()
        response = index.query(prompt + " respond in Chinese.",
                               text_qa_template=getQAPrompt(), similarity_top_k=15, response_mode="tree_summarize", mode=QueryMode.EMBEDDING, service_context=service_context, use_async=True)
        print(response)
        return str(response), 200
    else:
        return "index not found", 404


@ app.route("/index/stream", methods=["POST"])
def stream_index():
    q = Queue()
    id = request.json.get('id')
    prompt = request.json.get('prompt')

    app.logger.info(id)
    app.logger.info(prompt)

    cm = ChatOpenAI(
        streaming=True,
        openai_api_key=openai_key,
        temperature=0,
        callback_manager=CallbackManager([CustomCallBackHandler(queue=q)]), verbose=True)
    lp = LLMPredictor(llm=cm)

    sc = ServiceContext.from_defaults(
        llm_predictor=lp, prompt_helper=prompt_helper)
    index_file = os.path.join(Path('./data'), Path(id))

    if os.path.exists(index_file):
        mutex.acquire(timeout=10)
        index = GPTSimpleVectorIndex.load_from_disk(
            index_file, service_context=service_context)
        mutex.release()

        def query():
            resp = index.query(prompt + " respond in Chinese.",
                               text_qa_template=getQAPrompt(), similarity_top_k=10, response_mode="tree_summarize", mode=QueryMode.EMBEDDING, service_context=sc, use_async=True)
            q.put("[END]")
            print(resp)

        t = threading.Thread(target=query)
        t.start()

        def generate():
            while True:
                token = q.get()
                if token == "[END]":
                    print("stream end!")
                    break
                yield token

        return Response(generate(), mimetype='text/event-stream')

    else:
        return "index not found", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5601)
