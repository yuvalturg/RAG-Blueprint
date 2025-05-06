import uuid
import json
import streamlit as st
from llama_stack_client import Agent, AgentEventLogger
from llama_stack_client.types.agents.turn import ShieldCallStep
from modules.api import llama_stack_api
from modules.utils import data_url_from_file


def get_sidebar_config():
    """
    Gather and return sidebar configuration options.
    """
    # Select vector DBs
    dbs = llama_stack_api.client.vector_dbs.list()
    vector_db_ids = [db.identifier for db in dbs]
    selected_vector_dbs = st.multiselect(
        "Select Vector Databases", vector_db_ids,
        help="Choose one or more vector databases to search for relevant documents during RAG retrieval."
    )

    # Model selection
    models = [m.identifier for m in llama_stack_api.client.models.list() if m.model_type == "llm"]
    selected_model = st.selectbox(
        "Choose a model", models, index=0,
        help="Select the language model to use for generating chat responses."
    )

    def reset_agent():
        st.session_state.clear()
        st.cache_resource.clear()

    # MCP tool groups
    groups = [tg.identifier for tg in llama_stack_api.client.toolgroups.list() if tg.identifier.startswith("mcp::")]
    selected_toolgroups = st.multiselect(
        "MCP Servers", options=groups, default=[], on_change=reset_agent,
        help="Load tools from one or more MCP server groups. Clears chat when changed."
    )

    # Compute total tools count for display
    total = sum(len(llama_stack_api.client.tools.list(toolgroup_id=g)) for g in selected_toolgroups)
    if selected_vector_dbs:
        total += 1  # built-in RAG tool


    # Show active tools in an expander grouped by source
    with st.expander(f"Active Tools: 🛠 {total}", expanded=False):
        # list MCP server tools
        for group in selected_toolgroups:
            tools = llama_stack_api.client.tools.list(toolgroup_id=group)
            if tools:
                st.markdown(f"**{group}**")
                for t in tools:
                    st.write(f"- {t.identifier}")
        # list built-in RAG tool if any vector DB selected
        if selected_vector_dbs:
            st.markdown("**builtin::rag**")
            st.write("- knowledge_search")

    # Toggles and sliders
    tool_debug = st.toggle(
        "Tool Debug Messages", value=False,
        help="Enable detailed debug logs for every tool invocation."
    )
    temperature = st.slider(
        "Temperature", 0.0, 1.0, 0.0, 0.1,
        help="Controls randomness in responses: higher values produce more varied outputs."
    )
    top_p = st.slider(
        "Top P", 0.0, 1.0, 0.95, 0.05,
        help="Nucleus sampling cutoff: restricts next-token selection to the top probability mass."
    )
    max_tokens = st.slider(
        "Max Tokens", 0, 4096, 512, 1,
        help="Maximum number of tokens to generate in each response."
    )
    repetition_penalty = st.slider(
        "Repetition Penalty", 1.0, 2.0, 1.0, 0.1,
        help="Penalty applied to repeated tokens to reduce verbatim repetition."
    )

    # Shields selection
    shields = [s.identifier for s in llama_stack_api.client.shields.list()]
    input_shields = st.multiselect(
        "Input Shields", options=shields,
        help="Select input safety shields to validate or transform user inputs before processing."
    )
    output_shields = st.multiselect(
        "Output Shields", options=shields,
        help="Select output safety shields to filter or modify model outputs for safety."
    )

    stream = st.toggle(
        "Stream", value=True,
        help="Enable streaming of partial responses for real-time feedback."
    )
    system_prompt = st.text_area(
        "System Prompt", value="You are a helpful AI assistant.",
        help="Initial prompt defining the assistant's persona and behavior."
    )

    # Clear chat
    if st.button(
        "Clear Chat", use_container_width=True,
        help="Clear the current chat history and reset session state."
    ):
        st.session_state.messages = []
        st.rerun()

    return {
        'vector_dbs': selected_vector_dbs,
        'model': selected_model,
        'toolgroups': selected_toolgroups,
        'tool_debug': tool_debug,
        'temperature': temperature,
        'top_p': top_p,
        'max_tokens': max_tokens,
        'repetition_penalty': repetition_penalty,
        'input_shields': input_shields,
        'output_shields': output_shields,
        'stream': stream,
        'system_prompt': system_prompt,
    }


def render_history(tool_debug):
    """Render stored chat messages."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'debug_events' not in st.session_state:
        st.session_state.debug_events = []
    dbg_idx = 0
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
            # show debug events under each assistant message
            if tool_debug and msg['role'] == 'assistant' and dbg_idx < len(st.session_state.debug_events):
                with st.expander("Tool Debug Output", expanded=False):
                    for e in st.session_state.debug_events[dbg_idx]:
                        st.write(e)
                dbg_idx += 1


def create_agent_from_cfg(cfg):
    """Initialize and return an Agent based on sidebar options."""
    # Sampling strategy
    if cfg['temperature'] > 0:
        strategy = {'type': 'top_p', 'temperature': cfg['temperature'], 'top_p': cfg['top_p']}
    else:
        strategy = {'type': 'greedy'}
    sampling = {'strategy': strategy, 'max_tokens': cfg['max_tokens'], 'repetition_penalty': cfg['repetition_penalty']}

    # Assemble tools
    tools = list(cfg['toolgroups'])
    if cfg['vector_dbs']:
        tools.append({'name': 'builtin::rag/knowledge_search', 'args': {'vector_db_ids': cfg['vector_dbs']}})

    return Agent(
        llama_stack_api.client,
        model=cfg['model'],
        instructions=cfg['system_prompt'],
        sampling_params=sampling,
        tools=tools,
        input_shields=cfg['input_shields'],
        output_shields=cfg['output_shields'],
    )


def chat_page():
    """Main Chat page entry point."""
    st.title("💬 Chat")

    # Gather configuration and initialize
    with st.sidebar:
        cfg = get_sidebar_config()

    render_history(cfg['tool_debug'])
    # Initialize debug events storage
    if 'debug_events' not in st.session_state:
        st.session_state.debug_events = []

    agent = create_agent_from_cfg(cfg)
    session_id = agent.create_session(f"session-{uuid.uuid4()}")

    # Handle user input and response
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        # Render assistant reply
        with st.chat_message('assistant'):
            placeholder = st.empty()
            placeholder.markdown("...")
            full = ''
            tool_events = []
            # Send turn to llama-stack and get response
            response = agent.create_turn(
                messages=[{'role': 'user', 'content': prompt}],
                session_id=session_id,
                stream=cfg['stream']
            )
            # Streaming vs non-streaming
            if cfg['stream']:
                # Stream inference chunks inline, collecting tool and shield events separately
                for evt in AgentEventLogger().log(response):
                    role = getattr(evt, 'role', None)
                    if role == 'tool_execution':
                        tool_events.append(evt)
                    elif role == 'shield_call':
                        # extract only the user_message after the JSON metadata
                        end_meta = evt.content.rfind('}')
                        msg = evt.content[end_meta+1:].strip() if end_meta != -1 else evt.content
                        if(msg!="No Violation"):    
                            full += msg
                            placeholder.markdown(full + '▌')
                    else:
                        full += evt.content
                        placeholder.markdown(full + '▌')
                # finalize streamed content without cursor
                placeholder.markdown(full)
            else:
                # non-stream: response is a Turn
                turn = response
                # handle shields: if any shield_call triggered, show only its message
                shield_steps = [s for s in turn.steps if getattr(s, 'step_type', None) == 'shield_call' and s.violation]
                if shield_steps:
                    # use the last shield violation message
                    full = shield_steps[-1].violation.user_message
                else:
                    # collect tool events from steps
                    for step in turn.steps:
                        if getattr(step, 'step_type', None) == 'tool_execution':
                            for call in step.tool_calls:
                                tool_events.append(f"Tool:{call.tool_name} Args:{call.arguments}")
                            for resp in step.tool_responses:
                                tool_events.append(f"Tool:{resp.tool_name} Response:{resp.content}")
                    # final assistant output
                    full = turn.output_message.content
                # render response when not streaming
                placeholder.markdown(full)
            # Show tool debug output only when enabled
            if cfg['tool_debug']:
                with st.expander("Tool Debug Output"):
                    for evt in tool_events:
                        st.write(evt)
        # Save assistant message to history
        st.session_state.messages.append({'role': 'assistant', 'content': full})
        # Persist tool debug events across turns
        st.session_state.debug_events.append(tool_events)

chat_page()
