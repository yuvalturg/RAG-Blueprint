# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Inspect page: allows browsing and viewing details of Llama Stack resources.
"""
from streamlit_option_menu import option_menu
import streamlit as st

<<<<<<<< HEAD:frontend/llama_stack/distribution/ui/page/distribution/inspect.py
from page.distribution.datasets import datasets
from page.distribution.eval_tasks import benchmarks
from page.distribution.models import models
from page.distribution.scoring_functions import scoring_functions
from page.distribution.shields import shields
from page.distribution.vector_dbs import vector_dbs
from page.distribution.providers import providers
========
from llama_stack.distribution.ui.page.distribution.datasets import datasets
from llama_stack.distribution.ui.page.distribution.eval_tasks import benchmarks
from llama_stack.distribution.ui.page.distribution.models import models
from llama_stack.distribution.ui.page.distribution.scoring_functions import scoring_functions
from llama_stack.distribution.ui.page.distribution.shields import shields
from llama_stack.distribution.ui.page.distribution.vector_dbs import vector_dbs
>>>>>>>> upstream/main:frontend/llama_stack/distribution/ui/page/distribution/resources.py


def inspect_page():
    """
    Display a horizontal menu to select a resource and show its details.
    """
    st.header("🔍 Inspect")
    options = [
        "API Providers",
        "Models",
        "Vector Databases",
        "Shields",
        "Scoring Functions",
        "Datasets",
        "Benchmarks",
    ]
    icons = ["plug", "magic", "memory", "shield", "file-bar-graph", "database", "list-task"]
    selected_resource = option_menu(
        None,
        options,
        icons=icons,
        orientation="horizontal",
        styles={
            "nav-link": {
                "font-size": "12px",
            },
        },
    )
    if selected_resource == "Benchmarks":
        benchmarks()
    elif selected_resource == "Vector Databases":
        vector_dbs()
    elif selected_resource == "Datasets":
        datasets()
    elif selected_resource == "Models":
        models()
    elif selected_resource == "Scoring Functions":
        scoring_functions()
    elif selected_resource == "Shields":
        shields()
    elif selected_resource == "API Providers":
        providers()


inspect_page()
