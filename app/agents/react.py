"""Shared ReAct tool-calling loop.

Drives a `bind_tools` model together with a LangGraph `ToolNode`: the model may
call tools, read their results, and call again, until it answers without
requesting a tool (or a safety cap is reached). All three agents' research nodes
use this instead of each hand-rolling the loop.
"""

import logging

logger = logging.getLogger(__name__)


async def run_tool_loop(model, tool_node, messages: list, max_iters: int = 6) -> list:
    """Let `model` iteratively call tools via `tool_node` until it stops asking for
    tools or `max_iters` is hit. Every model reply and tool result is appended to
    `messages`, which is returned (the enriched conversation a final structured
    pass can summarize). The cap prevents a model that keeps calling tools from
    looping forever."""
    for _ in range(max_iters):
        response = await model.ainvoke(messages)
        messages.append(response)
        if not getattr(response, "tool_calls", None):
            return messages
        result = await tool_node.ainvoke({"messages": messages})
        messages.extend(result["messages"])
    logger.warning("run_tool_loop hit max_iters=%s without a final answer", max_iters)
    return messages
