"""Checkpoint manifest for <app>.

List the components this app needs; ``python -m wllm.weights <app>``
downloads whatever is missing. Reuse the shared components from
``wllm.weights.components`` where they apply (they download once and
are shared across apps); define app-specific components here. If a
component you define becomes shared by a second app, promote it to
``wllm/weights/components.py``.

Leave COMPONENTS empty if every model loads by HuggingFace repo id at
runtime.
"""

# from wllm.serving.weights.components import Component, WAN_TEXT_ENCODER, WAN_TOKENIZER
#
# MY_DIT = Component(
#     target="<model-dir>",
#     repo="<org/repo>",
#     patterns=("transformer/*",),
#     note="...",
# )

COMPONENTS = []
