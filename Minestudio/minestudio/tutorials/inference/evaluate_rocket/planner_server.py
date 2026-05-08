import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

import requests

from minestudio.tutorials.inference.evaluate_rocket.pointing_common import canonicalize_target_text


VALID_INTERACTIONS = {"Approach", "Interact", "Hunt", "Mine", "Craft", "Switch", "Use"}


def infer_target_and_final_interaction(task_text: str):
    task = task_text.lower().strip()
    if any(token in task for token in ["wood", "tree", "log", "oak"]):
        return "oak log", "Mine"
    if "sheep" in task:
        return "sheep", "Hunt"
    if "cow" in task:
        return "cow", "Hunt"
    if "zombie" in task:
        return "zombie", "Hunt"
    if "skeleton" in task:
        return "skeleton", "Hunt"
    if "spider" in task:
        return "spider", "Hunt"
    if "coal" in task:
        return "coal ore", "Mine"
    if "iron" in task:
        return "iron ore", "Mine"
    if "diamond" in task:
        return "diamond ore", "Mine"
    if "obsidian" in task:
        return "obsidian block", "Mine"
    if "chest" in task:
        return "chest", "Interact"
    if "bed" in task or "sleep" in task:
        return "bed", "Use"
    if "boat" in task:
        return "boat", "Use"
    if "table" in task or "crafting table" in task:
        return "crafting table", "Interact"
    if "door" in task or "gate" in task:
        return "door", "Interact"
    if "fire" in task:
        return "fire", "Use"

    words = re.findall(r"[a-zA-Z]+", task)
    if words:
        return " ".join(words[-2:]), "Approach"
    return "target object", "Approach"


def parse_json_object(text: str):
    text = text.strip()
    if not text:
        raise ValueError("empty planner response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Planner did not return valid JSON: {text}")


def normalize_interaction_type(interaction_type: str, fallback_interaction: str) -> str:
    if interaction_type not in VALID_INTERACTIONS:
        return fallback_interaction
    return interaction_type


def normalize_subtask(subtask: Dict[str, Any], fallback_target: str, fallback_interaction: str):
    target_text = canonicalize_target_text(subtask.get("target_text") or fallback_target)
    interaction_type = normalize_interaction_type(
        subtask.get("interaction_type") or fallback_interaction,
        fallback_interaction,
    )
    return {
        "target_text": target_text,
        "interaction_type": interaction_type,
        "success_criterion": subtask.get("success_criterion", ""),
        "reasoning": subtask.get("reasoning", ""),
    }


def normalize_plan_payload(payload, fallback_target: str, fallback_interaction: str):
    subtasks = payload.get("subtasks") or []
    normalized_subtasks = []
    for subtask in subtasks:
        if isinstance(subtask, dict):
            normalized_subtasks.append(normalize_subtask(subtask, fallback_target, fallback_interaction))

    if not normalized_subtasks:
        normalized_subtasks = [normalize_subtask(payload, fallback_target, fallback_interaction)]

    current_subtask_index = int(payload.get("current_subtask_index", 0) or 0)
    current_subtask_index = max(0, min(current_subtask_index, len(normalized_subtasks) - 1))
    active_subtask = normalized_subtasks[current_subtask_index]
    reasoning = payload.get("reasoning", "")
    done = bool(payload.get("done", False))
    subgoal_status = payload.get("subgoal_status", "unknown")
    planner_phase = payload.get("planner_phase", "review")
    advance_to_next_subtask = bool(payload.get("advance_to_next_subtask", False))
    return {
        "subtasks": normalized_subtasks,
        "current_subtask_index": current_subtask_index,
        "active_subtask": active_subtask,
        "active_subtask_reasoning": active_subtask.get("reasoning", ""),
        "target_text": active_subtask["target_text"],
        "interaction_type": active_subtask["interaction_type"],
        "reasoning": reasoning,
        "subgoal_status": subgoal_status,
        "advance_to_next_subtask": advance_to_next_subtask,
        "planner_phase": planner_phase,
        "done": done,
    }


def build_heuristic_subtasks(task_text: str):
    target_text, final_interaction = infer_target_and_final_interaction(task_text)
    target_text = canonicalize_target_text(target_text)
    subtasks: List[Dict[str, str]] = []
    if final_interaction != "Approach":
        subtasks.append(
            {
                "target_text": target_text,
                "interaction_type": "Approach",
                "success_criterion": f"The agent is close enough to directly {final_interaction.lower()} the target.",
                "reasoning": "Move close to the target first.",
            }
        )
    subtasks.append(
        {
            "target_text": target_text,
            "interaction_type": final_interaction,
            "success_criterion": f"The target has been successfully {final_interaction.lower()}ed or the task visibly progressed.",
            "reasoning": "Perform the task-specific final interaction.",
        }
    )
    return subtasks


def parse_last_plan(last_plan: str):
    if not last_plan:
        return {}
    if isinstance(last_plan, dict):
        return last_plan
    try:
        return json.loads(last_plan)
    except json.JSONDecodeError:
        return {}


class PlannerBackend:
    def __init__(
        self,
        backend: str,
        backend_url: Optional[str],
        model_id: str,
        api_key: str,
        timeout: int,
    ):
        self.backend = backend
        self.backend_url = backend_url.rstrip("/") if backend_url else None
        self.model_id = model_id
        self.api_key = api_key
        self.timeout = timeout
        self._finalize_openai_settings()

    def _finalize_openai_settings(self):
        if self.backend == "openai":
            self.backend = "openai-compatible"
            if not self.backend_url:
                self.backend_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            if self.api_key == "EMPTY":
                self.api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _normalize_content(self, content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_chunks = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_chunks.append(item.get("text", ""))
            return "\n".join(chunk for chunk in text_chunks if chunk).strip()
        return str(content)

    def _heuristic_plan(self, task_text: str, current_interaction: str, last_plan: str):
        fallback_target, fallback_interaction = infer_target_and_final_interaction(task_text)
        previous = parse_last_plan(last_plan)
        subtasks = previous.get("subtasks") or build_heuristic_subtasks(task_text)
        current_subtask_index = int(previous.get("current_subtask_index", 0) or 0)
        current_subtask_index = max(0, min(current_subtask_index, len(subtasks) - 1))
        if (
            current_subtask_index == 0
            and len(subtasks) > 1
            and current_interaction == "Approach"
        ):
            current_subtask_index = 1
            planner_phase = "review"
            subgoal_status = "ready_to_switch"
            reasoning = "Heuristic planner advanced from approach to the final interaction."
        else:
            planner_phase = "initial_plan" if not previous else "review"
            subgoal_status = "in_progress"
            reasoning = "Heuristic planner kept the current subtask."
        payload = {
            "subtasks": subtasks,
            "current_subtask_index": current_subtask_index,
            "reasoning": reasoning,
            "subgoal_status": subgoal_status,
            "planner_phase": planner_phase,
            "done": False,
        }
        return normalize_plan_payload(payload, fallback_target, fallback_interaction)

    def _openai_compatible_initial_plan(
        self,
        task_text: str,
        image_base64: str,
        model_id: Optional[str] = None,
    ):
        model_id = model_id or self.model_id
        if not self.backend_url:
            raise ValueError("--backend-url is required for openai-compatible backend")

        fallback_target, fallback_interaction = infer_target_and_final_interaction(task_text)
        prompt = f"""
You are a high-level planner for a Minecraft visuomotor controller.
Decompose the task into a short ordered list of visually grounded subtasks that can be executed by a low-level controller.

Allowed interaction_type values: Approach, Interact, Hunt, Mine, Craft, Switch, Use.

Task: {task_text}

Return strict JSON only:
{{
  "subtasks": [
    {{
      "target_text": "...",
      "interaction_type": "...",
      "success_criterion": "...",
      "reasoning": "Why this specific subtask is needed before later subtasks."
    }}
  ],
  "current_subtask_index": 0,
  "reasoning": "Explain briefly why this ordered subtask list makes sense given the current observation.",
  "done": false
}}

Rules:
- Produce between 1 and 6 subtasks.
- Each target_text must be a short visually-groundable noun phrase, not an instruction.
- Good target_text examples: "oak tree trunk", "sheep", "crafting table", "chest", "coal ore".
- Bad target_text examples: "collect wood", "go to the tree and mine it", "move closer to the sheep".
- If the task is long-horizon, break it into concrete ordered subtasks.
- If the first subtask requires movement, use interaction_type="Approach".
""".strip()

        response = requests.post(
            f"{self.backend_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": model_id,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                            },
                        ],
                    }
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload["choices"][0]["message"]["content"]
        parsed = parse_json_object(self._normalize_content(message))
        parsed["planner_phase"] = "initial_plan"
        parsed["subgoal_status"] = "in_progress"
        return normalize_plan_payload(parsed, fallback_target, fallback_interaction)

    def _openai_compatible_review_plan(
        self,
        task_text: str,
        image_base64: str,
        current_interaction: str,
        last_action_summary: str,
        plan_count: int,
        history: str,
        last_plan: str,
        model_id: Optional[str] = None,
    ):
        model_id = model_id or self.model_id
        if not self.backend_url:
            raise ValueError("--backend-url is required for openai-compatible backend")

        fallback_target, fallback_interaction = infer_target_and_final_interaction(task_text)
        previous = parse_last_plan(last_plan)
        subtasks = previous.get("subtasks") or build_heuristic_subtasks(task_text)
        current_subtask_index = int(previous.get("current_subtask_index", 0) or 0)
        current_subtask_index = max(0, min(current_subtask_index, len(subtasks) - 1))
        active_subtask = subtasks[current_subtask_index]
        remaining_subtasks = subtasks[current_subtask_index + 1 :]

        prompt = f"""
You are a high-level planner for a Minecraft visuomotor controller.
Review the current subtask and decide whether to keep working on it, advance to the next queued subtask, or finish the overall task.

Allowed interaction_type values: Approach, Interact, Hunt, Mine, Craft, Switch, Use.

Task: {task_text}
Current interaction: {current_interaction or "None"}
Current active subtask: {json.dumps(active_subtask, ensure_ascii=False)}
Remaining queued subtasks: {json.dumps(remaining_subtasks, ensure_ascii=False)}
Last planner output: {last_plan or "none"}
Last action summary: {last_action_summary or "none"}
Plan count: {plan_count}
Recent planner history: {history or "none"}

Return strict JSON only:
{{
  "target_text": "...",
  "interaction_type": "...",
  "subgoal_status": "in_progress|ready_to_switch|complete|stuck",
  "advance_to_next_subtask": false,
  "reasoning": "Explain what in the current observation/action history suggests the subtask is still in progress, ready to switch, complete, or stuck.",
  "done": false
}}

Rules:
- Keep the current queued subtask unless it is clearly wrong.
- If the current subtask is complete and there is another queued subtask, set advance_to_next_subtask=true.
- If the overall task is complete, set done=true.
- target_text must remain a short visually-groundable noun phrase.
- If movement is still needed, prefer interaction_type="Approach".
""".strip()

        response = requests.post(
            f"{self.backend_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": model_id,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                            },
                        ],
                    }
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload["choices"][0]["message"]["content"]
        parsed = parse_json_object(self._normalize_content(message))
        advance = bool(parsed.get("advance_to_next_subtask", False))
        updated_subtasks = list(subtasks)
        updated_subtasks[current_subtask_index] = normalize_subtask(parsed, active_subtask["target_text"], active_subtask["interaction_type"])
        if advance and current_subtask_index + 1 < len(updated_subtasks):
            current_subtask_index += 1
        parsed["subtasks"] = updated_subtasks
        parsed["current_subtask_index"] = current_subtask_index
        parsed["planner_phase"] = "review"
        parsed["advance_to_next_subtask"] = advance
        if parsed.get("done", False) and current_subtask_index >= len(updated_subtasks) - 1:
            parsed["subgoal_status"] = parsed.get("subgoal_status", "complete")
        return normalize_plan_payload(parsed, fallback_target, fallback_interaction)

    def plan(
        self,
        task_text: str,
        image_base64: str,
        current_interaction: str,
        last_action_summary: str,
        plan_count: int,
        history: str,
        last_plan: str,
        model_id: Optional[str] = None,
    ):
        if self.backend == "mock":
            subtasks = [
                {
                    "target_text": "oak tree trunk",
                    "interaction_type": "Approach",
                    "success_criterion": "The agent is close to the tree trunk.",
                    "reasoning": "Move close to the tree first.",
                },
                {
                    "target_text": "oak tree trunk",
                    "interaction_type": "Mine",
                    "success_criterion": "The wood has been collected.",
                    "reasoning": "Mine the tree trunk.",
                },
            ]
            current_subtask_index = 0 if plan_count <= 0 else min(1, len(subtasks) - 1)
            return normalize_plan_payload(
                {
                    "subtasks": subtasks,
                    "current_subtask_index": current_subtask_index,
                    "reasoning": "Mock planner queue.",
                    "subgoal_status": "in_progress" if current_subtask_index == 0 else "ready_to_switch",
                    "planner_phase": "initial_plan" if plan_count <= 0 else "review",
                    "done": False,
                },
                "oak tree trunk",
                "Mine",
            )
        if self.backend == "heuristic":
            return self._heuristic_plan(task_text, current_interaction, last_plan)
        if self.backend == "openai-compatible":
            previous = parse_last_plan(last_plan)
            if previous.get("subtasks"):
                return self._openai_compatible_review_plan(
                    task_text=task_text,
                    image_base64=image_base64,
                    current_interaction=current_interaction,
                    last_action_summary=last_action_summary,
                    plan_count=plan_count,
                    history=history,
                    last_plan=last_plan,
                    model_id=model_id,
                )
            return self._openai_compatible_initial_plan(
                task_text=task_text,
                image_base64=image_base64,
                model_id=model_id,
            )
        raise ValueError(f"Unsupported backend: {self.backend}")


def make_handler(backend: PlannerBackend):
    class PlannerHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "backend": backend.backend,
                        "model_id": backend.model_id,
                    },
                )
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/plan":
                self._send_json(404, {"error": "not found"})
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
                result = backend.plan(
                    task_text=payload["task_text"],
                    image_base64=payload["image_base64"],
                    current_interaction=payload.get("current_interaction", ""),
                    last_action_summary=payload.get("last_action_summary", ""),
                    plan_count=int(payload.get("plan_count", 0)),
                    history=payload.get("history", ""),
                    last_plan=payload.get("last_plan", ""),
                    model_id=payload.get("model_id"),
                )
                print(
                    "[planner-server] plan",
                    json.dumps(
                        {
                            "task_text": payload.get("task_text", ""),
                            "planner_phase": result.get("planner_phase", ""),
                            "current_interaction": payload.get("current_interaction", ""),
                            "plan_count": int(payload.get("plan_count", 0)),
                            "current_subtask_index": result.get("current_subtask_index", 0),
                            "num_subtasks": len(result.get("subtasks", [])),
                            "target_text": result.get("target_text", ""),
                            "interaction_type": result.get("interaction_type", ""),
                            "subgoal_status": result.get("subgoal_status", ""),
                            "active_subtask_reasoning": result.get("active_subtask_reasoning", ""),
                            "reasoning": result.get("reasoning", ""),
                            "done": result.get("done", False),
                        },
                        ensure_ascii=False,
                    ),
                )
                self._send_json(200, result)
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def log_message(self, fmt, *args):
            print(f"[planner-server] {self.address_string()} - {fmt % args}")

    return PlannerHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9164)
    parser.add_argument(
        "--backend",
        type=str,
        default="heuristic",
        choices=["mock", "heuristic", "openai-compatible", "openai"],
    )
    parser.add_argument("--backend-url", type=str, default="")
    parser.add_argument("--model-id", type=str, default="gpt-4o-mini")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    backend = PlannerBackend(
        backend=args.backend,
        backend_url=args.backend_url,
        model_id=args.model_id,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(backend))
    print(
        f"Planner server listening on http://{args.host}:{args.port} "
        f"(backend={args.backend}, model_id={args.model_id})"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
