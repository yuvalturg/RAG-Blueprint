# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Optional
import yaml
from llama_stack_client import LlamaStackClient
from modules.register_mcp_servers import RegisterMCPServers

"""
API wrapper module to initialize and expose a configured LlamaStackClient.
"""

class LlamaStackApi:
    """
    Encapsulates LlamaStackClient setup with environment-configured endpoints and API keys.
    Also registers MCP servers from YAML config.
    """
    def __init__(self):
        self.client = LlamaStackClient(
            base_url=os.environ.get("LLAMA_STACK_ENDPOINT", "http://localhost:8321"),
            
            provider_data={
                # Environment variables for various providers' API keys
                "fireworks_api_key": os.environ.get("FIREWORKS_API_KEY", ""),
                "together_api_key": os.environ.get("TOGETHER_API_KEY", ""),
                "sambanova_api_key": os.environ.get("SAMBANOVA_API_KEY", ""),
                "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
            },
        )
        RegisterMCPServers().register(self.client)

    def run_scoring(self, row, scoring_function_ids: list[str], scoring_params: Optional[dict]):
        """Run scoring for a single row, defaulting params if none provided."""
        if not scoring_params:
            scoring_params = {fn_id: None for fn_id in scoring_function_ids}
        return self.client.scoring.score(input_rows=[row], scoring_functions=scoring_params)

llama_stack_api = LlamaStackApi()
