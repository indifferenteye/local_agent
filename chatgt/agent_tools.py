#!/usr/bin/env python3

import json
from dataclasses import dataclass
from typing import Callable, Dict, List


ToolHandler = Callable[[Dict[str, object]], Dict[str, object]]
ToolFormatter = Callable[[Dict[str, object]], str]


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    example: Dict[str, object]
    handler: ToolHandler
    formatter: ToolFormatter | None = None


def default_formatter(observation: Dict[str, object]) -> str:
    return json.dumps(observation, indent=2, ensure_ascii=False)


def format_list_files(observation: Dict[str, object]) -> str:
    entries = observation.get("entries", [])

    if not isinstance(entries, list):
        return "I listed the path, but the result format was unexpected."

    if not entries:
        return "I can see the work directory, but it is empty."

    lines = ["I can see these files and folders:"]

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        name = entry.get("name", "")
        item_type = entry.get("type", "item")
        size = entry.get("size")

        if item_type == "directory":
            lines.append(f"- `{name}/` folder")
        elif size is None:
            lines.append(f"- `{name}` file")
        else:
            lines.append(f"- `{name}` file, {size} bytes")

    return "\n".join(lines)


def format_read_file(observation: Dict[str, object]) -> str:
    filename = observation.get("filename", "the file")
    content = observation.get("content", "")

    return f"Contents of `{filename}`:\n\n{content}"


def format_fetch_url(observation: Dict[str, object]) -> str:
    url = observation.get("final_url") or observation.get("url") or "the URL"
    status_code = observation.get("status_code", "unknown")
    content_type = observation.get("content_type", "")
    content = observation.get("content", "")
    content_type_suffix = f" and content type `{content_type}`" if content_type else ""

    return f"Fetched `{url}` with status {status_code}{content_type_suffix}.\n\n{content}"


def format_run_command(observation: Dict[str, object]) -> str:
    stdout = str(observation.get("stdout", "")).strip()
    stderr = str(observation.get("stderr", "")).strip()

    if stdout and stderr:
        return f"Command output:\n\n{stdout}\n\nErrors:\n\n{stderr}"

    if stdout:
        return f"Command output:\n\n{stdout}"

    if stderr:
        return f"Command produced errors:\n\n{stderr}"

    return "The command completed successfully with no output."


def format_browser_snapshot(observation: Dict[str, object]) -> str:
    title = observation.get("title", "")
    url = observation.get("url", "")
    text = observation.get("text", "")
    links = observation.get("links", [])
    buttons = observation.get("buttons", [])

    lines = [
        f"Browser page: {title or '(untitled)'}",
        f"URL: {url}",
        "",
        str(text).strip(),
    ]

    if isinstance(links, list) and links:
        lines.extend(["", "Links:"])
        for link in links[:10]:
            if isinstance(link, dict):
                lines.append(f"- {link.get('text', '')} -> {link.get('href', '')}")

    if isinstance(buttons, list) and buttons:
        lines.extend(["", "Buttons:"])
        for button in buttons[:10]:
            if isinstance(button, dict):
                lines.append(f"- {button.get('text', '')}")

    return "\n".join(lines).strip()


def format_browser_screenshot(observation: Dict[str, object]) -> str:
    filename = observation.get("filename", "browser-screenshot.png")
    url = observation.get("url", "")
    return f"Saved browser screenshot to `{filename}` from {url}."


def format_send_image(observation: Dict[str, object]) -> str:
    image = observation.get("image", {})

    if isinstance(image, dict):
        filename = image.get("filename", "the image")
        return f"Attached `{filename}` to the chat."

    return "Attached the image to the chat."


def build_default_tools(agent) -> Dict[str, AgentTool]:
    tools: List[AgentTool] = [
        AgentTool(
            name="respond",
            description=(
                "Use this for greetings, questions, explanations, creative writing, "
                "rewrites, continuations, or anything that does not require file/action work."
            ),
            example={
                "summary": "Short progress update for the user.",
                "action": "respond",
            },
            handler=lambda action: {
                "success": True,
                "message": str(action.get("message", "")),
            },
        ),
        AgentTool(
            name="list_files",
            description="List files and folders in the restricted work directory.",
            example={
                "summary": "Short progress update for the user.",
                "action": "list_files",
                "path": ".",
            },
            handler=lambda action: agent.list_files(str(action.get("path", "."))),
            formatter=format_list_files,
        ),
        AgentTool(
            name="read_file",
            description="Read a text file from the restricted work directory.",
            example={
                "summary": "Short progress update for the user.",
                "action": "read_file",
                "filename": "example.html",
            },
            handler=lambda action: agent.read_file(str(action.get("filename", ""))),
            formatter=format_read_file,
        ),
        AgentTool(
            name="write_file",
            description="Create or replace a text file in the restricted work directory.",
            example={
                "summary": "Short progress update for the user.",
                "action": "write_file",
                "filename": "example.html",
                "content": "complete file content here",
            },
            handler=lambda action: agent.write_file(
                str(action.get("filename", "")),
                agent.clean_file_content(str(action.get("content", ""))),
            ),
        ),
        AgentTool(
            name="run_command",
            description="Run a small allowlisted shell command in the restricted work directory.",
            example={
                "summary": "Short progress update for the user.",
                "action": "run_command",
                "command": "ls -la",
            },
            handler=lambda action: agent.run_command(str(action.get("command", ""))),
            formatter=format_run_command,
        ),
        AgentTool(
            name="fetch_url",
            description="Read public HTTP/HTTPS page content. Prefer this over curl or wget.",
            example={
                "summary": "Short progress update for the user.",
                "action": "fetch_url",
                "url": "https://example.com/",
            },
            handler=lambda action: agent.fetch_url(str(action.get("url", ""))),
            formatter=format_fetch_url,
        ),
        AgentTool(
            name="browser_open",
            description="Open JavaScript-rendered, visual, or interactive pages in Chromium.",
            example={
                "summary": "Short progress update for the user.",
                "action": "browser_open",
                "url": "https://example.com/",
            },
            handler=lambda action: agent.browser_open(str(action.get("url", ""))),
            formatter=format_browser_snapshot,
        ),
        AgentTool(
            name="browser_snapshot",
            description="Inspect the current browser page after opening, clicking, or typing.",
            example={
                "summary": "Short progress update for the user.",
                "action": "browser_snapshot",
            },
            handler=lambda action: agent.browser_snapshot(),
            formatter=format_browser_snapshot,
        ),
        AgentTool(
            name="browser_click",
            description="Click a visible element on the current browser page by CSS selector.",
            example={
                "summary": "Short progress update for the user.",
                "action": "browser_click",
                "selector": "button[type=submit]",
            },
            handler=lambda action: agent.browser_click(str(action.get("selector", ""))),
            formatter=format_browser_snapshot,
        ),
        AgentTool(
            name="browser_type",
            description="Fill an input or textarea on the current browser page by CSS selector.",
            example={
                "summary": "Short progress update for the user.",
                "action": "browser_type",
                "selector": "input[name=q]",
                "text": "search terms",
            },
            handler=lambda action: agent.browser_type(
                str(action.get("selector", "")),
                str(action.get("text", "")),
            ),
            formatter=format_browser_snapshot,
        ),
        AgentTool(
            name="browser_screenshot",
            description="Save a screenshot when visual layout matters or the user asks what a page looks like.",
            example={
                "summary": "Short progress update for the user.",
                "action": "browser_screenshot",
                "filename": "page.png",
            },
            handler=lambda action: agent.browser_screenshot(
                str(action.get("filename", "browser-screenshot.png"))
            ),
            formatter=format_browser_screenshot,
        ),
        AgentTool(
            name="browser_close",
            description="Close the browser when a browsing task is done.",
            example={
                "summary": "Short progress update for the user.",
                "action": "browser_close",
            },
            handler=lambda action: agent.browser_close(),
        ),
        AgentTool(
            name="send_image",
            description="Attach an existing image file from the work directory to the next chat response.",
            example={
                "summary": "Short progress update for the user.",
                "action": "send_image",
                "filename": "page.png",
                "label": "Screenshot",
            },
            handler=lambda action: agent.send_image(
                str(action.get("filename", "")),
                str(action.get("label", "")) or None,
            ),
            formatter=format_send_image,
        ),
        AgentTool(
            name="finish",
            description="Use this only after completing an agentic task.",
            example={
                "summary": "Short final summary for the user.",
                "action": "finish",
                "message": "Final response to the user.",
            },
            handler=lambda action: {
                "success": True,
                "finished": True,
                "message": str(action.get("message", action.get("summary", ""))),
            },
        ),
    ]

    return {tool.name: tool for tool in tools}


def render_tool_prompt(tools: Dict[str, AgentTool]) -> str:
    sections = []

    for index, tool in enumerate(tools.values(), start=1):
        example = json.dumps(tool.example, indent=2, ensure_ascii=False)
        sections.append(f"{index}. {tool.name}\n{tool.description}\n{example}")

    return "\n\n".join(sections)


def render_tool_rules() -> str:
    return """- For normal conversation, use respond.
- For questions that only need an answer, use respond.
- For coding/file tasks, work step by step.
- Prefer write_file for creating or editing files.
- After a successful write_file, usually finish immediately unless the user explicitly asked you to test, inspect, or refine the result.
- Do not rewrite the same file repeatedly unless the previous observation showed an error.
- Do not use shell redirection, heredocs, pipes, backticks, ampersands, or destructive commands.
- For website or URL content, use fetch_url instead of run_command with curl or wget.
- For JavaScript-rendered pages, visual inspection, forms, buttons, or navigation, use browser_open and browser_snapshot.
- Use browser_screenshot when the user asks what a page looks like or when visual layout matters.
- Use send_image after creating or finding an image file that should be shown in the chat."""
