# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import enum
import json
import uuid

import streamlit as st
from llama_stack_client import Agent, AgentEventLogger
from llama_stack_client.lib.agents.react.agent import ReActAgent
from llama_stack_client.lib.agents.react.tool_parser import ReActOutput
from llama_stack_client.types import ChatCompletionResponse # Not explicitly used, but good for type hinting or future use

from llama_stack.distribution.ui.modules.api import llama_stack_api


class AgentType(enum.Enum):
    """Defines the types of agents available for chat."""
    REGULAR = "Regular"
    REACT = "ReAct"

def reset_agent_and_chat():
    """Clears Streamlit session state and cached resources.
    Typically used when configurations change to ensure a fresh state.
    """
    st.session_state.clear()
    st.cache_resource.clear()

def get_strategy(temperature, top_p):
    """Determines the sampling strategy for the LLM based on temperature."""
    return {'type': 'greedy'} if temperature == 0 else {
            'type': 'top_p', 'temperature': temperature, 'top_p': top_p
        }

def render_history(tool_debug):
    """Renders the chat history from session state.
    Also displays debug events for assistant messages if tool_debug is enabled.
    """
    # Initialize messages in session state if not present
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    # Initialize debug_events in session state if not present
    if 'debug_events' not in st.session_state:
         st.session_state.debug_events = []

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

            # Display debug events expander for assistant messages (excluding the initial greeting)
            if msg['role'] == 'assistant' and tool_debug and i > 0:
                # Debug events are stored per assistant turn.
                # The index for debug_events corresponds to the assistant message turn.
                # messages: [A_initial, U_1, A_1, U_2, A_2, ...]
                # debug_events: [events_for_A_1, events_for_A_2, ...]
                # For A_1 (msg index 2), debug_events index is (2//2)-1 = 0.
                debug_event_list_index = (i // 2) - 1
                if 0 <= debug_event_list_index < len(st.session_state.debug_events):
                    current_turn_events_list = st.session_state.debug_events[debug_event_list_index]

                    if current_turn_events_list: # Only show expander if there are events
                        with st.expander("Tool/Debug Events", expanded=False):
                            if isinstance(current_turn_events_list, list) and len(current_turn_events_list) > 0:
                                for event_idx, event_item in enumerate(current_turn_events_list):
                                    with st.container():
                                        if isinstance(event_item, dict):
                                            st.json(event_item, expanded=False)
                                        elif isinstance(event_item, str):
                                            st.text_area(
                                                label=f"Debug Event {event_idx + 1}",
                                                value=event_item,
                                                height=100,
                                                disabled=True,
                                                key=f"debug_event_msg{i}_item{event_idx}" # Unique key for each text area
                                            )
                                        else:
                                            st.write(event_item) # Fallback for other data types
                                        if event_idx < len(current_turn_events_list) - 1:
                                            st.divider()
                            elif isinstance(current_turn_events_list, list) and not current_turn_events_list:
                                st.caption("No debug events recorded for this turn.")
                            else: # Should not happen with current logic
                                st.write("Debug data for this turn (unexpected format):")
                                st.write(current_turn_events_list)


def tool_chat_page():
    """Main function to render the Streamlit chat page and handle its logic."""
    st.title("💬 Chat")

    client = llama_stack_api.client
    models = client.models.list()
    model_list = [model.identifier for model in models if hasattr(model, 'identifier') and model.api_model_type == "llm"]
    if not model_list:
        st.error("No LLM models found. Please check Llama Stack server configuration.")
        st.stop()

    tool_groups = client.toolgroups.list()
    tool_groups_list = [tg.identifier for tg in tool_groups if hasattr(tg, 'identifier')]
    mcp_tools_list = [tool for tool in tool_groups_list if tool.startswith("mcp::")]
    builtin_tools_list = [tool for tool in tool_groups_list if not tool.startswith("mcp::")]

    selected_vector_dbs = [] # Initialize before sidebar

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.title("Configuration")

        processing_mode = st.radio(
            "Processing mode",
            ["Agent-based", "Direct"],
            index=0, # Default to Agent-based
            captions=[
                "Uses an Agent (Regular or ReAct) with tools.",
                "Directly calls the model with optional RAG.",
            ],
            on_change=reset_agent_and_chat,
            help="Choose how requests are processed. 'Direct' bypasses agents, 'Agent-based' uses them.",
        )

        if processing_mode == "Agent-based":
            st.subheader("Agent Type")
            agent_type = st.radio(
                "Select Agent Type",
                [AgentType.REGULAR, AgentType.REACT],
                format_func=lambda x: x.value,
                help="Choose the agent logic: Regular (simpler) or ReAct (step-by-step reasoning)."
            )
        else:
            agent_type = None # Not applicable for Direct mode

        st.subheader("Model")
        model = st.selectbox(
            label="Model",
            options=model_list,
            on_change=reset_agent_and_chat,
            label_visibility="collapsed",
            index=0 if model_list else -1
        )

        # Tool selection for Agent-based mode
        selected_builtin_tools = []
        if processing_mode == "Agent-based":
            st.subheader("Available ToolGroups")
            selected_builtin_tools = st.multiselect(
                label="Built-in tools",
                options=builtin_tools_list,
                on_change=reset_agent_and_chat,
                format_func=lambda tool: "".join(tool.split("::")[1:]),
                help="Select built-in tools for the agent.",
            )

        # RAG Vector DB Selection (available in Direct mode or if 'builtin::rag' is selected in Agent mode)
        rag_enabled_in_agent = processing_mode == "Agent-based" and "builtin::rag" in selected_builtin_tools
        if processing_mode == "Direct" or rag_enabled_in_agent:
            vector_dbs_available = llama_stack_api.client.vector_dbs.list() or []
            if vector_dbs_available:
                 vector_db_options = [vector_db.identifier for vector_db in vector_dbs_available if hasattr(vector_db, 'identifier')]
                 if vector_db_options:
                     selected_vector_dbs = st.multiselect(
                         label="Select Vector Databases (for RAG)",
                         options=vector_db_options,
                         on_change=reset_agent_and_chat,
                         help="Choose collections for RAG retrieval."
                     )
                 else:
                      st.caption("No vector databases found.")
            else:
                 st.caption("No vector databases available.")

        selected_mcp_tools = []
        if processing_mode == "Agent-based":
            selected_mcp_tools = st.multiselect(
                label="MCP Servers",
                options=mcp_tools_list,
                default=[],
                on_change=reset_agent_and_chat,
                format_func=lambda tool: "".join(tool.split("::")[1:]),
                help="Select tools hosted on MCP servers.",
            )

        agent_toolgroup_selection = []
        if processing_mode == "Agent-based":
            agent_toolgroup_selection = selected_builtin_tools + selected_mcp_tools

        # Display Active Tools/Features
        total_tools = 0
        active_tools_display = {}
        if processing_mode == "Agent-based":
            for toolgroup_id_or_dict in agent_toolgroup_selection:
                # The RAG tool might be a dict if vector DBs are configured for it.
                current_group_id = toolgroup_id_or_dict['name'] if isinstance(toolgroup_id_or_dict, dict) else toolgroup_id_or_dict
                try:
                    tools = client.tools.list(toolgroup_id=current_group_id)
                    tool_ids = [tool.identifier for tool in tools if hasattr(tool, 'identifier')]
                    if tool_ids:
                        active_tools_display[current_group_id] = tool_ids
                    total_tools += len(tool_ids)
                except Exception as e:
                    st.warning(f"Could not list tools for {current_group_id}: {e}")
        elif processing_mode == "Direct" and selected_vector_dbs:
             active_tools_display["builtin::rag"] = ["knowledge_search (Direct Mode with RAG)"]
             total_tools += 1 # Count RAG as an active feature

        with st.expander(f"Active Tools/Features: {total_tools}", expanded=False):
             if not active_tools_display:
                 st.caption("No tools or RAG databases selected.")
             else:
                 for group_id, tools_in_group in active_tools_display.items():
                     st.markdown(f"**{group_id.split('::')[-1] if '::' in group_id else group_id}**") # Cleaner display name
                     for tool_id_in_group in tools_in_group:
                          clean_tool_name = tool_id_in_group.split(':')[-1] if ':' in tool_id_in_group else tool_id_in_group
                          st.write(f"- {clean_tool_name}")

        input_shields = []
        output_shields = []
        if processing_mode == "Agent-based":
            st.subheader("Security Shields")
            shields_available = client.shields.list()
            shield_options = [s.identifier for s in shields_available if hasattr(s, 'identifier')]
            input_shields = st.multiselect("Input Shields", options=shield_options, on_change=reset_agent_and_chat)
            output_shields = st.multiselect("Output Shields", options=shield_options, on_change=reset_agent_and_chat)

        st.subheader("Sampling Parameters")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.1, 0.05, on_change=reset_agent_and_chat)
        top_p = st.slider("Top P", 0.0, 1.0, 0.95, 0.05, on_change=reset_agent_and_chat)
        max_tokens = st.slider("Max Tokens", 1, 4096, 512, 64, on_change=reset_agent_and_chat)
        repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.0, 0.05, on_change=reset_agent_and_chat)

        st.subheader("System Prompt")
        default_prompt = "You are a helpful AI assistant."
        if processing_mode == "Agent-based" and agent_type == AgentType.REACT:
            default_prompt = "You are a helpful ReAct agent. Reason step-by-step to fulfill the user query using available tools."
        system_prompt = st.text_area(
            "System Prompt", value=default_prompt, on_change=reset_agent_and_chat, height=100
        )

        st.subheader("Response Handling")
        stream_opt = st.toggle("Stream Response", value=True, on_change=reset_agent_and_chat)
        tool_debug = st.toggle("Show Tool/Debug Info", value=False)

        if st.button("Clear Chat & Reset Config", use_container_width=True):
            reset_agent_and_chat()
            st.rerun()

    # Configure RAG tool with selected vector databases for agent mode
    # This modifies `agent_toolgroup_selection` in place if "builtin::rag" is selected
    if processing_mode == "Agent-based":
        for i, tool_name in enumerate(agent_toolgroup_selection):
            if tool_name == "builtin::rag":
                agent_toolgroup_selection[i] = dict(
                    name="builtin::rag",
                    args={"vector_db_ids": list(selected_vector_dbs)},
                )

    # --- Agent Creation (Not cached here, but managed in session state later) ---
    def create_agent_instance():
        """Creates an agent instance based on current sidebar configurations."""
        sampling_strategy = get_strategy(temperature, top_p)
        sampling_params = {
            'strategy': sampling_strategy,
            'max_tokens': max_tokens,
            'repetition_penalty': repetition_penalty,
        }
        common_agent_args = {
            'client': client,
            'model': model,
            'tools': agent_toolgroup_selection, # This now includes configured RAG
            'sampling_params': sampling_params,
            'instructions': system_prompt,
            'input_shields': input_shields,
            'output_shields': output_shields,
        }

        if agent_type == AgentType.REACT:
            react_response_format = None
            try:
                 react_response_format = {
                     "type": "json_schema",
                     "json_schema": ReActOutput.model_json_schema(),
                 }
            except AttributeError:
                st.warning("ReActOutput schema not found; ReAct agent might not parse output correctly.")
            return ReActAgent(**common_agent_args, response_format=react_response_format)
        else: # Regular Agent
            return Agent(**common_agent_args)

    # --- Unified Response Handler Functions (Streaming and Non-Streaming) ---
    # These functions process the response from the LLM or Agent and yield/return content.
    # They also populate `debug_events_list` for display.

    def _handle_direct_response(response, placeholder, debug_events_list):
        """Handles non-streaming direct API responses."""
        assistant_response_content = ""
        try:
            completion_msg = getattr(response, 'completion_message', None)
            if completion_msg:
                assistant_response_content = getattr(completion_msg, 'content', "")
            placeholder.markdown(assistant_response_content)
        except Exception as e:
            error_msg = f"Error processing direct response: {e}"
            st.error(error_msg)
            placeholder.markdown(f"An error occurred: {e}")
            debug_events_list.append({"type": "error", "source": "_handle_direct_response", "content": str(e)})
            assistant_response_content = f"Error: {str(e)}" # For state
        return assistant_response_content

    def _handle_regular_agent_response(response_turn, placeholder, debug_events_list):
        """Handles non-streaming regular agent responses (expects a Turn object)."""
        assistant_response_content = ""
        try:
            output_msg = getattr(response_turn, 'output_message', None)
            if output_msg and hasattr(output_msg, 'content'):
                assistant_response_content = getattr(output_msg, 'content', "[Content missing]")
                debug_events_list.append({"type": "regular_agent_final_output", "content_preview": assistant_response_content[:100]})
            else:
                assistant_response_content = "[No output_message or content found in agent turn]"
                debug_events_list.append({"type": "error", "source": "_handle_regular_agent_response", "details": "output_message missing"})
            placeholder.markdown(assistant_response_content)
        except Exception as e:
            error_msg = f"Error processing regular agent non-streaming response: {e}"
            st.error(error_msg)
            placeholder.markdown(f"🚨 **Error:** {error_msg}")
            debug_events_list.append({"type": "error", "source": "_handle_regular_agent_response", "content": str(e)})
            assistant_response_content = f"Error: {str(e)}"
        return assistant_response_content

    def _process_inference_step_non_streaming(current_step_content_json_str: str):
        """
        Processes the JSON content of an inference step for non-streaming display.
        Returns a list of formatted markdown strings for the step and the final answer if present.
        """
        formatted_parts = []
        final_answer_found = None
        try:
            react_output_data = json.loads(current_step_content_json_str)
            thought = react_output_data.get("thought")
            action_data = react_output_data.get("action") # Renamed to avoid conflict with 'action' key if any
            answer = react_output_data.get("answer")

            if thought:
                # Using HTML <details> for collapsibility within Markdown
                formatted_parts.append(f"<details><summary>🤔 **Thinking**</summary>\n\n```text\n{thought}\n```\n\n</details>")

            if action_data and isinstance(action_data, dict):
                tool_name = action_data.get("tool_name")
                tool_params = action_data.get("tool_params", {})
                formatted_parts.append(
                    f"<details><summary>🛠️ **Action:** Using tool `{tool_name}`</summary>\n\n**Parameters:**\n```json\n{json.dumps(tool_params, indent=2)}\n```\n\n</details>"
                )

            if answer and answer != "null" and answer is not None:
                final_answer_found = str(answer) # Ensure it's a string
        except json.JSONDecodeError:
            formatted_parts.append(f"⚠️ **Error:** Failed to parse ReAct step content as JSON.\nContent:\n```text\n{current_step_content_json_str}\n```")
        except Exception as e:
            formatted_parts.append(f"⚠️ **Error:** Failed to process ReAct inference step: {e}\nContent:\n```text\n{current_step_content_json_str}\n```")
        return formatted_parts, final_answer_found

    def _process_tool_execution_non_streaming(step_details, collected_tool_results: list):
        """
        Processes tool execution details for non-streaming display.
        Returns a list of formatted markdown strings for observations.
        Appends (tool_name, content_str) to collected_tool_results.
        """
        formatted_parts = []
        try:
            if hasattr(step_details, "tool_responses") and step_details.tool_responses:
                for tool_response in step_details.tool_responses:
                    tool_name = getattr(tool_response, 'tool_name', 'Unknown Tool')
                    content_str = getattr(tool_response, 'content', 'No content')
                    # Store raw content string for potential later summary
                    collected_tool_results.append((tool_name, content_str))

                    observation_summary = f"⚙️ **Observation (Result from `{tool_name}`):**"
                    try:
                        # Try to parse and pretty-print JSON, otherwise show raw content
                        parsed_content = json.loads(content_str)
                        observation_details = f"\n\n```json\n{json.dumps(parsed_content, indent=2)}\n```\n\n"
                    except json.JSONDecodeError:
                        observation_details = f"\n\n```text\n{content_str}\n```\n\n" # Show as raw text if not JSON

                    formatted_parts.append(f"<details><summary>{observation_summary}</summary>{observation_details}</details>")
            else:
                formatted_parts.append("<details><summary>⚙️ **Observation**</summary>\n\nTool execution completed, but no response data found.\n\n</details>")
        except Exception as e:
            formatted_parts.append(f"⚠️ **Error:** Error processing tool execution details: {str(e)}")
        return formatted_parts # collected_tool_results is modified by appending

    def _format_tool_results_summary_non_streaming(tool_results: list):
        """Formats a summary of tool results for non-streaming display, returning a single Markdown string."""
        if not tool_results:
            return ""

        summary_parts = ["\n\n#### Summary of Information Gathered by Tools:"]
        for tool_name, content_str in tool_results:
            try:
                # Basic summary: just mention the tool and that it provided data.
                # More sophisticated parsing (like in the streaming version) can be added here if needed.
                summary_parts.append(f"* **`{tool_name}`** provided information (see observation details above).")

            except Exception as e: # Catch any error during summarization of this specific tool's result
                summary_parts.append(f"* Error summarizing result from `{tool_name}`: {str(e)}")

        if len(summary_parts) == 1: # Only the header
            return "" # No useful summary to show
        return "\n".join(summary_parts)


    # --- Main Non-Streaming ReAct Handler ---
    def _handle_react_response(turn_object, placeholder, debug_events_list):
        """
        Handles non-streaming ReAct agent responses from a Turn object.
        Collects all thoughts, actions, observations, and the final answer,
        then displays them all at once. 'turn_object' is expected to be a 'Turn' instance.
        """
        # Main container for all Markdown parts
        full_response_md_accumulator = ["### ReAct Agent Processing Log"]
        current_step_raw_llm_output = "" # Accumulates text from step_progress for an inference step
        final_answer_from_steps = None
        accumulated_tool_results_for_summary = [] # Stores (tool_name, content_str) for summary

        try:
            if not hasattr(turn_object, 'events') or not turn_object.events:
                final_md_output = "Agent turn completed with no detailed events."
                placeholder.markdown(final_md_output)
                debug_events_list.append({"type": "info", "source": "_handle_react_response", "content": "No events in Turn object."})
                return final_md_output

            # Loop through each event in the Turn object
            for agent_event_index, agent_event in enumerate(turn_object.events):
                payload = getattr(agent_event, 'payload', None)
                if not payload:
                    error_detail = f"Malformed agent event (missing payload) at index {agent_event_index}: {agent_event}"
                    full_response_md_accumulator.append(f"\n\n---\n⚠️ **Internal Error:**\n{error_detail}")
                    debug_events_list.append({"type": "error", "source": "_handle_react_response_event_parsing", "content": error_detail})
                    continue

                event_type = getattr(payload, 'event_type', None)
                delta = getattr(payload, 'delta', None) # AgentEventDelta
                step_details = getattr(payload, 'step_details', None) # AgentStepDetails

                debug_events_list.append({
                    "type": "react_event_non_stream_processed",
                    "event_index": agent_event_index,
                    "event_type": event_type,
                    "payload_type": type(payload).__name__,
                    "has_delta": delta is not None,
                    "has_step_details": step_details is not None,
                })

                # Accumulate LLM output for the current inference step
                if event_type == "step_progress" and delta and getattr(delta, "type", None) == "text":
                    current_step_raw_llm_output += getattr(delta, 'text', '')

                # Process completed steps (inference or tool execution)
                elif event_type == "step_complete" and step_details:
                    step_type = getattr(step_details, 'step_type', None)
                    full_response_md_accumulator.append(f"\n\n---\n#### Step: {str(step_type).upper() if step_type else 'UNKNOWN STEP TYPE'}")

                    if step_type == "inference":
                        if current_step_raw_llm_output:
                            # Process the accumulated LLM output for this inference step
                            inference_formatted_parts, answer_candidate = _process_inference_step_non_streaming(current_step_raw_llm_output)
                            full_response_md_accumulator.extend(inference_formatted_parts)
                            if answer_candidate: # Prioritize the latest answer found
                                final_answer_from_steps = answer_candidate
                            current_step_raw_llm_output = "" # Reset for the next potential inference step
                        else:
                            full_response_md_accumulator.append("\n_(This inference step did not produce direct text output via step_progress events. It might represent a structural change or decision.)_")

                    elif step_type == "tool_execution":
                        # Process tool execution details
                        tool_exec_formatted_parts = _process_tool_execution_non_streaming(
                            step_details, accumulated_tool_results_for_summary # Pass list to append to
                        )
                        full_response_md_accumulator.extend(tool_exec_formatted_parts)
                    else:
                        full_response_md_accumulator.append(f"\n_(Completed step of unhandled type: {step_type})_")
                        current_step_raw_llm_output = "" # Reset just in case

                elif event_type == "turn_complete":
                    # The turn_complete event might contain a final output message from the agent.
                    if hasattr(payload, 'output_message') and payload.output_message and hasattr(payload.output_message, 'content'):
                        turn_final_content = payload.output_message.content
                        if turn_final_content and not final_answer_from_steps: # Use if no step-based answer found
                            final_answer_from_steps = turn_final_content
                            debug_events_list.append({"type": "info", "source": "_handle_react_response", "content": "Used agent's turn_complete output_message as final answer."})
                    full_response_md_accumulator.append("\n\n---\n**Agent Turn Completed**")


                elif event_type == "error": # Explicit error event from the agent's turn processing
                    error_content = getattr(payload, 'message', 'Unknown error occurred within the agent turn.')
                    full_response_md_accumulator.append(f"\n\n---\n🛑 **Agent Error Reported:**\n{error_content}")
                    debug_events_list.append({"type": "error", "source": "react_agent_turn_event_error", "content": error_content})


            # After iterating through all events in the Turn object:
            # Assemble the final response message.
            full_response_md_accumulator.append("\n\n---\n### Final Outcome")

            # If tools were used and no specific "answer" was extracted, provide a summary of tool findings.
            if accumulated_tool_results_for_summary and not final_answer_from_steps:
                tool_summary_md = _format_tool_results_summary_non_streaming(accumulated_tool_results_for_summary)
                if tool_summary_md:
                    full_response_md_accumulator.append(tool_summary_md)

            # Add the final answer to the display
            if final_answer_from_steps:
                full_response_md_accumulator.append(f"\n\n🏁 **Final Answer:**\n\n{final_answer_from_steps}")
            elif accumulated_tool_results_for_summary: # Tools ran, but no explicit "answer"
                full_response_md_accumulator.append("\n\n🏁 **Conclusion:** The process involved the steps and tool uses detailed above. Please review the gathered information.")
            elif len(full_response_md_accumulator) <= 1: # Only the initial title, no other content
                # Check if the Turn object's top-level output_message has content (e.g., for very simple agent responses)
                direct_turn_output = getattr(getattr(turn_object, 'output_message', None), 'content', None)
                if direct_turn_output:
                    full_response_md_accumulator.append(f"\n\n💬 **Agent Response:**\n\n{direct_turn_output}")
                else:
                    full_response_md_accumulator.append("\nThe agent processed the request, but no detailed step-by-step breakdown or explicit final answer was generated in this format.")


            # Join all collected Markdown parts into a single string
            final_markdown_output = "\n".join(filter(None, full_response_md_accumulator))

            if not final_markdown_output.strip() or final_markdown_output == "### ReAct Agent Processing Log": # Handle empty or only title case
                final_markdown_output = "The ReAct agent processed the request, but no specific output details were generated for display."

            placeholder.markdown(final_markdown_output) # Display everything at once
            return final_markdown_output # Return the full content for chat history

        except Exception as e:
            import traceback
            st.error(f"Fatal error in _handle_react_response: {type(e).__name__}: {e}")
            error_message_for_display = f"🚨 **FATAL ERROR PROCESSING AGENT RESPONSE:**\n\n```\n{type(e).__name__}: {e}\n\n{traceback.format_exc()}\n```"
            placeholder.markdown(error_message_for_display)
            debug_events_list.append({
                "type": "error", "source": "_handle_react_response_fatal",
                "error_type": type(e).__name__, "message": str(e),
                "traceback": traceback.format_exc()
            })
            # Try to return some info from the turn object if possible for history
            raw_turn_info_preview = str(turn_object)[:500] # Limit size for history
            return f"Error processing agent response. Details: {e}. Raw turn info (partial): {raw_turn_info_preview}"


    def _handle_direct_stream_response(response_stream, placeholder, debug_events_list):
        """Handles streaming direct API responses, yielding text deltas."""
        try:
            for chunk in response_stream:
                event = getattr(chunk, 'event', None)
                delta = getattr(event, 'delta', None)
                if delta and getattr(delta, "type") == "text":
                    text_delta = getattr(delta, 'text', None)
                    if text_delta:
                        yield text_delta
                elif delta: # Other delta types for debugging
                    debug_events_list.append({"type": "stream_delta_other", "content": delta})
        except Exception as e:
            error_msg = f"Error processing direct stream: {e}"
            st.error(error_msg)
            placeholder.markdown(f"Stream error: {e}") # Show error in placeholder
            debug_events_list.append({"type": "error", "source": "_handle_direct_stream_response", "content": str(e)})
            yield f" Error in stream: {e}" # Yield error to be part of response content

    def _handle_regular_agent_stream_response(response_stream, placeholder, debug_events_list):
        """Handles streaming regular agent responses, yielding text deltas and tool usage."""
        try:
            # Use itertools.tee to duplicate the stream for UI and debug logging
            # This is crucial because a generator can only be consumed once.
            from itertools import tee
            ui_stream, debug_log_stream = tee(response_stream, 2)

            for response_event in ui_stream: # Iterate over the UI stream
                payload = getattr(getattr(response_event, 'event', None), 'payload', None)
                if payload:
                    event_type = getattr(payload, 'event_type', None)
                    delta = getattr(payload, 'delta', None)
                    step_details = getattr(payload, 'step_details', None)

                    if event_type == "step_progress" and delta and getattr(delta, "type") == "text":
                        placeholder.empty() # Clear "Thinking..." before streaming actual content
                        yield delta.text
                    elif event_type == "step_complete" and step_details:
                        if getattr(step_details, 'step_type', '') == "tool_execution":
                            if getattr(step_details, 'tool_calls', None):
                                tool_name = str(getattr(step_details.tool_calls[0], 'tool_name', 'Unknown Tool'))
                                yield f'\n\n🛠️ :grey[_Using "{tool_name}" tool..._]\n\n'
                            # Potentially add observation from tool_responses here if available and desired
                else: # Fallback for unexpected event structure
                    yield f" Received unexpected event from agent: {response_event}"
                    debug_events_list.append({"type": "warning", "source": "_handle_regular_agent_stream", "details": "Unexpected event structure", "event": str(response_event)[:200]})

            # Process the debug log stream separately
            # AgentEventLogger helps parse and structure these events
            for log_entry in AgentEventLogger().log(debug_log_stream):
                if log_entry.role == "tool_execution": # Or other relevant roles
                    debug_events_list.append({"type": "tool_log", "content": log_entry.content})
                # Add other log types as needed for debugging

        except Exception as e:
            error_msg = f"Error processing regular agent stream: {e}"
            st.error(error_msg)
            placeholder.markdown(f"Stream error: {e}")
            debug_events_list.append({"type": "error", "source": "_handle_regular_agent_stream_response", "content": str(e)})


    def _process_inference_step(current_step_content, tool_results, final_answer):
        try:
            react_output_data = json.loads(current_step_content)
            thought = react_output_data.get("thought")
            action = react_output_data.get("action")
            answer = react_output_data.get("answer")

            if answer and answer != "null" and answer is not None:
                final_answer = answer

            if thought:
                with st.expander("🤔 Thinking...", expanded=False):
                    st.markdown(f":grey[__{thought}__]")

            if action and isinstance(action, dict):
                tool_name = action.get("tool_name")
                tool_params = action.get("tool_params")
                with st.expander(f'🛠 Action: Using tool "{tool_name}"', expanded=False):
                    st.json(tool_params)

            if answer and answer != "null" and answer is not None:
                yield f"\n\n✅ **Final Answer:**\n{answer}"

        except json.JSONDecodeError:
            yield f"\n\nFailed to parse ReAct step content:\n```json\n{current_step_content}\n```"
        except Exception as e:
            yield f"\n\nFailed to process ReAct step: {e}\n```json\n{current_step_content}\n```"

        return final_answer

    def _process_tool_execution(step_details, tool_results):
        try:
            if hasattr(step_details, "tool_responses") and step_details.tool_responses:
                for tool_response in step_details.tool_responses:
                    tool_name = tool_response.tool_name
                    content = tool_response.content
                    tool_results.append((tool_name, content))
                    with st.expander(f'⚙️ Observation (Result from "{tool_name}")', expanded=False):
                        try:
                            parsed_content = json.loads(content)
                            st.json(parsed_content)
                        except json.JSONDecodeError:
                            st.code(content, language=None)
            else:
                with st.expander("⚙️ Observation", expanded=False):
                    st.markdown(":grey[_Tool execution step completed, but no response data found._]")
        except Exception as e:
            with st.expander("⚙️ Error in Tool Execution", expanded=False):
                st.markdown(f":red[_Error processing tool execution: {str(e)}_]")

        return tool_results

    def _format_tool_results_summary(tool_results):
        yield "\n\n**Here's what I found:**\n"
        for tool_name, content in tool_results:
            try:
                parsed_content = json.loads(content)

                if tool_name == "web_search" and "top_k" in parsed_content:
                    yield from _format_web_search_results(parsed_content)
                elif "results" in parsed_content and isinstance(parsed_content["results"], list):
                    yield from _format_results_list(parsed_content["results"])
                elif isinstance(parsed_content, dict) and len(parsed_content) > 0:
                    yield from _format_dict_results(parsed_content)
                elif isinstance(parsed_content, list) and len(parsed_content) > 0:
                    yield from _format_list_results(parsed_content)
            except json.JSONDecodeError:
                yield f"\n**{tool_name}** was used but returned complex data. Check the observation for details.\n"
            except (TypeError, AttributeError, KeyError, IndexError) as e:
                print(f"Error processing {tool_name} result: {type(e).__name__}: {e}")

    def _format_web_search_results(parsed_content):
        for i, result in enumerate(parsed_content["top_k"], 1):
            if i <= 3:
                title = result.get("title", "Untitled")
                url = result.get("url", "")
                content_text = result.get("content", "").strip()
                yield f"\n- **{title}**\n  {content_text}\n  [Source]({url})\n"

    def _format_results_list(results):
        for i, result in enumerate(results, 1):
            if i <= 3:
                if isinstance(result, dict):
                    name = result.get("name", result.get("title", "Result " + str(i)))
                    description = result.get("description", result.get("content", result.get("summary", "")))
                    yield f"\n- **{name}**\n  {description}\n"
                else:
                    yield f"\n- {result}\n"

    def _format_dict_results(parsed_content):
        yield "\n```\n"
        for key, value in list(parsed_content.items())[:5]:
            if isinstance(value, str) and len(value) < 100:
                yield f"{key}: {value}\n"
            else:
                yield f"{key}: [Complex data]\n"
        yield "```\n"

    def _format_list_results(parsed_content):
        yield "\n"
        for _, item in enumerate(parsed_content[:3], 1):
            if isinstance(item, str):
                yield f"- {item}\n"
            elif isinstance(item, dict) and "text" in item:
                yield f"- {item['text']}\n"
            elif isinstance(item, dict) and len(item) > 0:
                first_value = next(iter(item.values()))
                if isinstance(first_value, str) and len(first_value) < 100:
                    yield f"- {first_value}\n"
    def _handle_react_stream_response(turn_response, msg_placeholder, current_turn_debug_events):
        current_step_content = ""
        final_answer = None
        tool_results = []
        try:
            for response in turn_response:
                if not hasattr(response.event, "payload"):
                    yield (
                        "\n\n🚨 :red[_Llama Stack server Error:_]\n"
                        "The response received is missing an expected `payload` attribute.\n"
                        "This could indicate a malformed response or an internal issue within the server.\n\n"
                        f"Error details: {response}"
                    )
                    return

                payload = response.event.payload

                if payload.event_type == "step_progress" and hasattr(payload.delta, "text"):
                    current_step_content += payload.delta.text
                    continue

                if payload.event_type == "step_complete":
                    step_details = payload.step_details

                    if step_details.step_type == "inference":
                        yield from _process_inference_step(current_step_content, tool_results, final_answer)
                        current_step_content = ""
                    elif step_details.step_type == "tool_execution":
                        tool_results = _process_tool_execution(step_details, tool_results)
                        current_step_content = ""
                    else:
                        current_step_content = ""
        except Exception as e:
            error_msg = f"Fatal error processing ReAct stream: {e}"
            st.error(error_msg)
            yield f"\n\n🚨 **FATAL STREAM PROCESSING ERROR:** {error_msg}\n" # Yield to UI

        if not final_answer and tool_results:
            yield from _format_tool_results_summary(tool_results)


    def _augment_with_rag(prompt, selected_vector_dbs_for_rag, debug_events_list):
        """Augments the user prompt with RAG context if vector DBs are selected (for Direct mode)."""
        final_prompt_to_llm = prompt
        if selected_vector_dbs_for_rag:
            with st.spinner("Retrieving context (RAG)..."):
                try:
                    rag_response = client.tool_runtime.rag_tool.query(
                        content=prompt, vector_db_ids=list(selected_vector_dbs_for_rag)
                    )
                    rag_context = rag_response.content
                    debug_events_list.append({
                        "type": "rag_query_direct_mode", "query": prompt,
                        "vector_dbs": selected_vector_dbs_for_rag,
                        "context_length": len(rag_context) if rag_context else 0,
                        "context_preview": (str(rag_context[:200]) + "..." if rag_context else "None")
                    })

                    if rag_context:
                        final_prompt_to_llm = f"Based on the following context:\n\n{rag_context}\n\nPlease answer the query: {prompt}"
                        st.caption("ℹ️ Context retrieved via RAG and prepended to query.")
                    else:
                        st.caption("ℹ️ No relevant context found via RAG for this query.")
                except Exception as e:
                    st.warning(f"RAG Error (Direct Mode): {e}")
                    debug_events_list.append({"type": "error", "source": "rag_direct_mode", "content": str(e)})
        return final_prompt_to_llm

    def _handle_request(current_agent, current_session_id, user_prompt, msg_placeholder, current_turn_debug_events, stream_enabled, current_processing_mode, current_agent_type_for_handler, **direct_mode_specific_args):
        """Unified request handler for both Agent-based and Direct modes.
        Returns the content that should be stored in the assistant's message in session state.
        """
        assistant_response_content_for_history = "[Response not captured]" # Default

        try:
            if current_processing_mode == "Agent-based":
                if not current_agent:
                    st.error("Agent not initialized for Agent-based mode.")
                    return "Error: Agent not initialized."

                # Agent makes a turn
                agent_response_obj_or_stream = current_agent.create_turn(
                    session_id=current_session_id,
                    messages=[{"role": "user", "content": user_prompt}], # User prompt is the input
                    stream=stream_enabled,
                )

                if current_agent_type_for_handler == AgentType.REACT:
                    if stream_enabled:
                        assistant_response_content_for_history = st.write_stream(
                            _handle_react_stream_response(agent_response_obj_or_stream, msg_placeholder, current_turn_debug_events)
                        )
                    else: # Non-streaming ReAct
                        assistant_response_content_for_history = _handle_react_response(agent_response_obj_or_stream, msg_placeholder, current_turn_debug_events)
                else: # Regular Agent
                    if stream_enabled:
                        assistant_response_content_for_history = st.write_stream(
                            _handle_regular_agent_stream_response(agent_response_obj_or_stream, msg_placeholder, current_turn_debug_events)
                        )
                    else: # Non-streaming Regular Agent
                        assistant_response_content_for_history = _handle_regular_agent_response(agent_response_obj_or_stream, msg_placeholder, current_turn_debug_events)

            elif current_processing_mode == "Direct":
                # Extract necessary args for direct call from direct_mode_specific_args
                llm_model_id = direct_mode_specific_args.get('model')
                messages_for_api_call = direct_mode_specific_args.get('messages_for_api') # Already includes RAG augmentation
                sampling_params_for_api = direct_mode_specific_args.get('sampling_params')
                api_client = direct_mode_specific_args.get('client')

                if not all([llm_model_id, messages_for_api_call, sampling_params_for_api, api_client]):
                    st.error("Missing parameters for Direct API call.")
                    return "Error: Missing Direct API parameters."

                # Direct LLM call
                llm_response_obj_or_stream = api_client.inference.chat_completion(
                    model_id=llm_model_id,
                    messages=messages_for_api_call,
                    stream=stream_enabled,
                    sampling_params=sampling_params_for_api
                )

                if stream_enabled:
                    # st.write_stream consumes the generator and returns the concatenated string
                    final_streamed_content = st.write_stream(
                        _handle_direct_stream_response(llm_response_obj_or_stream, msg_placeholder, current_turn_debug_events)
                    )
                    assistant_response_content_for_history = final_streamed_content if final_streamed_content else "[Streamed Direct Response]"
                else: # Non-streaming Direct
                    assistant_response_content_for_history = _handle_direct_response(llm_response_obj_or_stream, msg_placeholder, current_turn_debug_events)
            else:
                st.error(f"Unknown processing mode: {current_processing_mode}")
                assistant_response_content_for_history = f"Error: Unknown mode {current_processing_mode}"

        except Exception as e:
            error_details = f"Error in _handle_request ({current_processing_mode}): {e}"
            st.error(error_details)
            msg_placeholder.markdown(f"An error occurred: {str(e)}") # Display in UI
            assistant_response_content_for_history = f"Error: {str(e)}" # Store error in history
            current_turn_debug_events.append({"type": "error", "source": "_handle_request_main", "details": str(e)})

        return assistant_response_content_for_history


    # --- Main Chat Logic ---
    # Initialize session state keys if they don't exist
    if "messages" not in st.session_state: # Chat history
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?", "stop_reason": "end_of_turn"}]
    if "debug_events" not in st.session_state: # Per-turn debug logs
        st.session_state["debug_events"] = []
    # `agent` and `agent_session_id_for_agent` will be managed based on config changes

    render_history(tool_debug) # Display current chat history and any past debug events

    # Prepare agent or direct mode parameters based on current config
    # This section ensures that an agent is created/reused appropriately
    # or that direct mode parameters are ready.
    agent_instance_for_turn = None
    agent_session_id_for_turn = None

    if processing_mode == "Agent-based":
        if agent_type is None: # Should be set if Agent-based is chosen
            st.warning("Please select an Agent Type in the sidebar to proceed with Agent-based mode.")
            st.stop()

        # Prepare a hashable representation of the RAG tool configuration for the key
        processed_agent_toolgroups_for_config_key = []
        raw_tool_selection = selected_builtin_tools + selected_mcp_tools # From sidebar
        for tool_group_name_from_selection in raw_tool_selection:
            if tool_group_name_from_selection == "builtin::rag":
                if selected_vector_dbs: # Only include RAG if DBs are selected
                    # Ensure consistent hashing for RAG config: sorted tuple of DB IDs
                    hashable_rag_item = (
                        "CONFIGURED_RAG_TOOL", # Marker
                        "builtin::rag",
                        tuple(sorted(list(selected_vector_dbs)))
                    )
                    processed_agent_toolgroups_for_config_key.append(hashable_rag_item)
                # If "builtin::rag" is selected but no vector_dbs, it's effectively omitted from agent config.
            else: # Other tool groups are just strings
                processed_agent_toolgroups_for_config_key.append(tool_group_name_from_selection)

        # Define a configuration key based on all relevant sidebar settings
        # This key determines if the existing agent in session_state is still valid.
        current_agent_config_key = (
            model, agent_type, tuple(processed_agent_toolgroups_for_config_key), # Includes RAG config
            # `selected_vector_dbs` is part of RAG config above, not needed separately here for agent key.
            max_tokens, system_prompt,
            tuple(sorted(input_shields)), tuple(sorted(output_shields)), # Sorted for consistency
            temperature, top_p, repetition_penalty
        )

        # Check if a valid agent for the current configuration already exists in session state
        if ('agent' in st.session_state and
            'agent_config_key' in st.session_state and
            st.session_state.agent_config_key == current_agent_config_key): # Direct comparison of tuples
            agent_instance_for_turn = st.session_state.agent
            agent_session_id_for_turn = st.session_state.agent_session_id_for_agent
            # Ensure `st.session_state.agent_type` (used by _handle_request) is current
            st.session_state.current_active_agent_type = agent_type
        else:
            # Configuration changed or no agent exists; create a new one.
            with st.spinner(f"Initializing {agent_type.value} Agent..."):
                try:
                    agent_instance_for_turn = create_agent_instance() # Uses current sidebar settings
                    st.session_state.agent = agent_instance_for_turn
                    st.session_state.agent_config_key = current_agent_config_key
                    st.session_state.current_active_agent_type = agent_type # Store type of created agent

                    # Create a new session for this specific agent instance
                    agent_session_id_for_turn = agent_instance_for_turn.create_session(
                        session_name=f"streamlit_chat_session_{uuid.uuid4()}"
                    )
                    st.session_state.agent_session_id_for_agent = agent_session_id_for_turn
                except Exception as e:
                    st.error(f"Failed to initialize agent: {e}")
                    # Clean up potentially partial state to avoid issues on next run
                    for key_to_del in ['agent', 'agent_config_key', 'agent_session_id_for_agent', 'current_active_agent_type']:
                        if key_to_del in st.session_state:
                            del st.session_state[key_to_del]
                    st.stop() # Halt execution if agent creation fails

    # Handle user chat input
    if user_prompt := st.chat_input("Ask a question..."):
        # Append user message to history and display it
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Prepare for assistant's response
        # Each assistant turn gets its own list for debug events
        st.session_state.debug_events.append([])
        current_turn_debug_events_list = st.session_state.debug_events[-1] # Get the list for this turn

        with st.chat_message("assistant"):
            message_placeholder = st.empty() # For streaming or final response
            message_placeholder.markdown("🤔 Thinking...") # Initial feedback

            direct_mode_handler_args = {} # Will be populated if in Direct mode
            prompt_for_processing = user_prompt # By default, use the raw user prompt

            if processing_mode == "Direct":
                # For Direct mode, augment prompt with RAG if vector DBs are selected
                if selected_vector_dbs:
                    prompt_for_processing = _augment_with_rag(
                        user_prompt, selected_vector_dbs, current_turn_debug_events_list
                    )
                # Else, prompt_for_processing remains the original user_prompt

                # Construct messages for direct API call (system + history + current user prompt)
                history_for_api = [msg for msg in st.session_state.get('messages', [])[:-1] if msg['role'] != 'assistant' or 'content' in msg] # Exclude user's current message, ensure assistant messages have content
                messages_for_direct_api = (
                    [{'role': 'system', 'content': system_prompt}] +
                    history_for_api +
                    [{'role': 'user', 'content': prompt_for_processing}]
                )
                sampling_params_direct = {
                    "strategy": get_strategy(temperature, top_p),
                    "max_tokens": max_tokens,
                    "repetition_penalty": repetition_penalty,
                }
                direct_mode_handler_args = {
                     "model": model,
                     "messages_for_api": messages_for_direct_api,
                     "sampling_params": sampling_params_direct,
                     "client": client # The API client instance
                }

            # Call the unified request handler
            # `agent_instance_for_turn` and `agent_session_id_for_turn` will be None if not in Agent-based mode.
            # `prompt_for_processing` is the (potentially RAG-augmented) prompt for Direct mode,
            # or the raw user_prompt for Agent mode (agent handles RAG internally if configured).
            assistant_response_final_content = _handle_request(
                current_agent=agent_instance_for_turn,
                current_session_id=agent_session_id_for_turn,
                user_prompt=user_prompt, # Agent mode gets raw prompt; Direct mode's RAG already handled
                msg_placeholder=message_placeholder,
                current_turn_debug_events=current_turn_debug_events_list,
                stream_enabled=stream_opt,
                current_processing_mode=processing_mode,
                current_agent_type_for_handler=st.session_state.get("current_active_agent_type"), # Type of the *active* agent
                **direct_mode_handler_args # Pass model, messages, etc., for Direct mode
            )

        # Update session state with the assistant's final response content
        st.session_state.messages.append({
            'role': 'assistant',
            "stop_reason": "end_of_turn", # Or other reason if available
            'content': assistant_response_final_content if assistant_response_final_content is not None else "[No response content captured]"
        })

        # Rerun to clear the input box and reflect the new message in history
        st.rerun()

# --- Entry point for the Streamlit page ---
if __name__ == "__main__":
    tool_chat_page()