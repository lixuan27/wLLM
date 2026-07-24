# SPDX-License-Identifier: Apache-2.0

from wllm.serving.configs.models.dits.longlive import LongLiveConfig
from wllm.serving.models.dit.krea_realtime import KreaRealtimeTransformer3DModel


class LongLiveTransformer3DModel(KreaRealtimeTransformer3DModel):
    _default_config = LongLiveConfig()

    def __init__(self, config: LongLiveConfig) -> None:
        super().__init__(config=config)


EntryClass = LongLiveTransformer3DModel
