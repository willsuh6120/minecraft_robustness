import re
import os
import json
import cv2
import time
from pathlib import Path
import argparse
import requests
import gradio as gr
import torch
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw

from minestudio.benchmark import prepare_task_configs
from minestudio.tutorials.inference.evaluate_rocket.utils import Session, Pointer, Planner

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), 
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 255, 255), (0, 0, 0), (128, 128, 128),
    (128, 0, 0), (128, 128, 0), (0, 128, 0),
    (128, 0, 128), (0, 128, 128), (0, 0, 128),
]

SEGMENT_MAPPING = {
    "Hunt": 0, "Use": 3, "Mine": 2, "Interact": 3, "Craft": 4, "Switch": 5, "Approach": 6
}

NOOP_ACTION = {
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
}

def reset_fn(env_name, world_seed, session):
    image = session.reset(env_name, world_seed)
    return image, f"Reset env=`{env_name}` seed={session.world_seed}", session

def step_fn(act_key, session):
    action = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in NOOP_ACTION.items()
    }
    if act_key not in (None, "null"):
        action[act_key] = 1
    policy_action = session.env.action_transformer.env2policy(action)
    policy_action = {
        "buttons": np.expand_dims(policy_action["buttons"], axis=0),
        "camera": np.expand_dims(policy_action["camera"], axis=0),
    }
    agent_action = session.env.action_mapper.from_factored(policy_action)
    image = session.step(agent_action)
    return image, session

def loop_step_fn(steps, session):
    for i in range(steps):
        image = session.step()
        status = f"Running Agent `Rocket` steps: {i+1}/{steps}. Last action: {session.last_action_summary}"
        yield image, session.num_steps, status, session

def staged_loop_step_fn(approach_steps, final_steps, reset_memory_between_phases, session):
    final_type = session.segment_type
    phase_plan = []
    if final_type != "Approach" and approach_steps > 0:
        phase_plan.append(("Approach", int(approach_steps)))
    if final_steps > 0 or not phase_plan:
        phase_plan.append((final_type, int(final_steps)))

    original_type = final_type
    try:
        for phase_idx, (phase_type, phase_steps) in enumerate(phase_plan):
            if phase_steps <= 0:
                continue
            session.segment_type = phase_type
            if phase_idx > 0 and reset_memory_between_phases:
                session.clear_agent_memory(reset_counters=False)
            for step_idx in range(phase_steps):
                image = session.step()
                status = (
                    f"Stage `{phase_type}` {step_idx+1}/{phase_steps}. "
                    f"Last action: {session.last_action_summary}"
                )
                yield image, session.num_steps, status, session
    finally:
        session.segment_type = original_type

def clear_memory_fn(session):
    image = session.current_image
    session.clear_agent_memory()
    return image, "0", session

def get_points_with_draw(image, label, session, evt: gr.SelectData):
    points = session.points
    point_label = session.points_label
    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 5, (0, 255, 0) if label == 'Add Points' else (255, 0, 0)
    points.append([x, y])
    point_label.append(1 if label == 'Add Points' else 0)
    image = np.copy(image)
    cv2.circle(image, (x, y), point_radius, point_color, -1)
    return image, session

def clear_points_fn(session):
    session.clear_points()
    return session.current_image, session

def segment_fn(session):
    if len(session.points) == 0:
        return session.current_image, session
    session.segment()
    session.clear_agent_memory()
    image = session.apply_mask()
    return image, session

def clear_segment_fn(session):
    session.clear_obj_mask()
    session.tracking_flag = False
    return session.current_image, False, session

def set_tracking_mode(tracking_flag, session):
    session.tracking_flag = tracking_flag
    return session

def set_segment_type(segment_type, session):
    session.segment_type = segment_type
    return session

def play_fn(session):
    image = session.step()
    return image, session

memory_length = gr.Textbox(value="0", interactive=False, show_label=False)

def make_video_fn(session, make_video, save_video, progress=gr.Progress()):
    images = session.image_history
    if len(images) == 0:
        return session, make_video, save_video
    filepath = "rocket.mp4"
    h, w = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filepath, fourcc, 20.0, (w, h))
    for image in progress.tqdm(images):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)
    video.release()
    session.image_history = []
    return session, gr.Button("Make Video", visible=False), gr.DownloadButton("Download!", value=filepath, visible=True)

def save_video_fn(session, make_video, save_video):
    return session, gr.Button("Make Video", visible=True), gr.DownloadButton("Download!", visible=False)

def choose_sam_fn(sam_choice, session):
    session.sam_choice = sam_choice
    session.load_sam()
    return session

def molmo_fn(molmo_text, molmo_session, rocket_session, display_image):
    image = rocket_session.current_image.copy()
    points = molmo_session.gen_point(image=image, prompt=molmo_text)
    molmo_result = molmo_session.molmo_result
    display_image = np.copy(display_image)
    rocket_session.clear_points()
    for x, y in points:
        x, y = int(x), int(y)
        point_radius, point_color = 5, (0, 255, 0) 
        rocket_session.points.append([x, y])
        rocket_session.points_label.append(1)
        cv2.circle(display_image, (x, y), point_radius, point_color, -1)
    return molmo_result, display_image

def planner_point_fn(task_text, planner_session, molmo_session, rocket_session, display_image):
    image = rocket_session.current_image.copy()
    history = json.dumps(rocket_session.plan_history[-4:], ensure_ascii=False)
    plan = planner_session.plan(
        task_text=task_text,
        image=image,
        current_interaction=rocket_session.segment_type,
        last_action_summary=rocket_session.last_action_summary,
        plan_count=rocket_session.plan_count,
        history=history,
        last_plan=rocket_session.last_plan,
    )
    rocket_session.plan_count += 1
    rocket_session.last_plan = plan
    rocket_session.plan_history.append(plan)
    target_text = plan.get("target_text", "")
    interaction_type = plan.get("interaction_type", "Approach")
    rocket_session.segment_type = interaction_type
    current_subtask_index = int(plan.get("current_subtask_index", 0) or 0)
    num_subtasks = len(plan.get("subtasks", []))
    active_subtask_reasoning = (plan.get("active_subtask_reasoning") or "").strip()
    planner_reasoning = (plan.get("reasoning") or "").strip()

    points = molmo_session.gen_point(image=image, prompt=target_text) if target_text else []
    planner_output = json.dumps(plan, ensure_ascii=False)
    display_image = np.copy(display_image)
    rocket_session.clear_points()
    for x, y in points:
        x, y = int(x), int(y)
        point_radius, point_color = 5, (0, 255, 0)
        rocket_session.points.append([x, y])
        rocket_session.points_label.append(1)
        cv2.circle(display_image, (x, y), point_radius, point_color, -1)
    status = (
        f"PLAN phase={plan.get('planner_phase', 'initial_plan')} "
        f"subtask={current_subtask_index + 1}/{max(1, num_subtasks)} "
        f"target={target_text or 'none'} "
        f"interaction={interaction_type} "
        f"subgoal_status={plan.get('subgoal_status', 'unknown')} "
        f"reasoning={(planner_reasoning or active_subtask_reasoning)[:180]}"
    )
    return planner_output, target_text, interaction_type, display_image, status

def planned_loop_step_fn(task_text, total_steps, replan_interval, reset_memory_on_replan, planner_session, molmo_session, session):
    replan_interval = max(1, int(replan_interval))
    current_target_text = session.last_plan.get("target_text", "")
    current_interaction = session.last_plan.get("interaction_type", session.segment_type)
    planner_output = json.dumps(session.last_plan, ensure_ascii=False) if session.last_plan else ""
    current_subgoal_status = session.last_plan.get("subgoal_status", "") if session.last_plan else ""
    current_subtask_index = int(session.last_plan.get("current_subtask_index", 0) or 0) if session.last_plan else 0
    current_num_subtasks = len(session.last_plan.get("subtasks", [])) if session.last_plan else 0
    current_active_subtask_reasoning = (session.last_plan.get("active_subtask_reasoning", "") if session.last_plan else "").strip()
    current_planner_reasoning = (session.last_plan.get("reasoning", "") if session.last_plan else "").strip()

    for i in range(int(total_steps)):
        should_replan = (i % replan_interval == 0) or not session.last_plan
        replan_message = ""
        if should_replan:
            history = json.dumps(session.plan_history[-4:], ensure_ascii=False)
            plan = planner_session.plan(
                task_text=task_text,
                image=session.current_image.copy(),
                current_interaction=session.segment_type,
                last_action_summary=session.last_action_summary,
                plan_count=session.plan_count,
                history=history,
                last_plan=session.last_plan,
            )
            session.plan_count += 1
            session.last_plan = plan
            session.plan_history.append(plan)
            current_target_text = plan.get("target_text", "")
            current_interaction = plan.get("interaction_type", "Approach")
            current_subgoal_status = plan.get("subgoal_status", "")
            current_subtask_index = int(plan.get("current_subtask_index", 0) or 0)
            current_num_subtasks = len(plan.get("subtasks", []))
            current_active_subtask_reasoning = (plan.get("active_subtask_reasoning") or "").strip()
            current_planner_reasoning = (plan.get("reasoning") or "").strip()
            session.segment_type = current_interaction
            planner_output = json.dumps(plan, ensure_ascii=False)
            replan_message = (
                f"REPLAN phase={plan.get('planner_phase', 'review')} "
                f"subtask={current_subtask_index + 1}/{max(1, current_num_subtasks)} "
                f"target={current_target_text or 'none'} "
                f"interaction={current_interaction} "
                f"subgoal_status={current_subgoal_status or 'unknown'} "
                f"reasoning={(current_planner_reasoning or current_active_subtask_reasoning)[:180]}. "
            )
            print(f"[planner-loop] step={i+1} {replan_message}{planner_output}")

            points = molmo_session.gen_point(image=session.current_image.copy(), prompt=current_target_text) if current_target_text else []
            session.clear_points()
            for x, y in points:
                session.points.append([int(x), int(y)])
                session.points_label.append(1)
            if points:
                session.segment()
                if reset_memory_on_replan:
                    session.clear_agent_memory(reset_counters=False)

        image = session.step()
        status = (
            f"{replan_message}Planned run {i+1}/{int(total_steps)}. "
            f"subtask={current_subtask_index + 1}/{max(1, current_num_subtasks)}. "
            f"phase={current_interaction}, target={current_target_text or 'none'}. "
            f"Last action: {session.last_action_summary}"
        )
        yield image, session.num_steps, status, planner_output, current_target_text, current_interaction, session

def extract_points(data):
    pattern = r'x\d?="([-+]?\d*\.\d+|\d+)" y\d?="([-+]?\d*\.\d+|\d+)"'
    points = re.findall(pattern, data)
    points = [(float(x)/100*640, float(y)/100*360) for x, y in points]
    return points


def draw_gradio_components(args):

    with gr.Blocks() as demo:
        
        gr.Markdown(
            """
            # Welcome to Explore ROCKET-1 in Minecraft!!
            ## Please follow next steps to interact with the agent:
            1. Reset the environment by selecting an environment name.
            2. Select a SAM2 checkpoint to load.
            3. Use your mouse to add or remove points on the image. 
            4. Select the segment type you want to perform.
            5. Enable `tracking` mode if you want to track objects while stepping actions. 
            6. Click `New Segment` to segment the image based on the points you added. 
            7. Call the agent by clicking `Call Rocket` to run the agent for a certain number of steps. 
            ## Hints:
            1. You can use the `Make Video` button to generate a video of the agent's actions.
            2. You can use the `Clear Memory` button to clear the ROCKET-1's memory. 
            3. You can use the `Clear Segment` button to clear SAM's memory. 
            4. You can use the `Manually Step` button to manually step the agent. 
            """
        )
        
        file_list = prepare_task_configs(args.task_group, path=args.task_group_path)
        name_file_mapping = {name: file for name, file in file_list.items()}
        env_list = [name for name, file in file_list.items()]
        
        rocket_session = gr.State(Session(
            model_path=args.model_path,
            sam_path=args.sam_path,
            name_file_mapping=name_file_mapping, 
        ))
        molmo_session = gr.State(Pointer(
            model_id=args.molmo_id,
            model_url=args.molmo_url,
            api_key=args.molmo_api_key,
        ))
        planner_session = gr.State(Planner(
            model_id=args.planner_id,
            model_url=args.planner_url,
            api_key=args.planner_api_key,
        ))
        with gr.Row():
            
            with gr.Column(scale=2):
                # start_image = Image.open("start.png").resize((640, 360))
                start_image = np.zeros((360, 640, 3), dtype=np.uint8)
                
                with gr.Group():
                    display_image = gr.Image(
                        value=np.array(start_image), 
                        interactive=False, 
                        show_label=False, 
                        label="Real-time Environment Observation", 
                        streaming=True
                    )
                    display_status = gr.Textbox("Status Bar", interactive=False, show_label=False)
            
            with gr.Column(scale=1):
                
                sam_choice = gr.Radio(
                    choices=["large", "base", "small", "tiny"],
                    value="base",
                    label="Select SAM2 checkpoint",
                )
                sam_choice.select(fn=choose_sam_fn, inputs=[sam_choice, rocket_session], outputs=[rocket_session], show_progress=False)
                
                with gr.Group():
                    add_or_remove = gr.Radio(
                        choices=["Add Points", "Remove Areas"], 
                        value="Add Points", 
                        label="Use you mouse to add or remove points",
                    )
                    clear_points_btn = gr.Button("Clear Points")
                    clear_points_btn.click(clear_points_fn, inputs=[rocket_session], outputs=[display_image, rocket_session], show_progress=True)
                
                with gr.Group():
                    segment_type = gr.Radio(
                        choices=["Approach", "Use", "Interact", "Hunt", "Mine", "Craft", "Switch"],
                        value="Approach", 
                        label="What do you want with this segment?",
                    )
                    track_flag = gr.Checkbox(True, label="Enable tracking objects while steping actions")
                    track_flag.select(fn=set_tracking_mode, inputs=[track_flag, rocket_session], outputs=[rocket_session], show_progress=False)
                    with gr.Group(), gr.Row():
                        new_segment_btn = gr.Button("New Segment")
                        clear_segment_btn = gr.Button("Clear Segment")
                        new_segment_btn.click(segment_fn, inputs=[rocket_session], outputs=[display_image, rocket_session], show_progress=True)
                        clear_segment_btn.click(clear_segment_fn, inputs=[rocket_session], outputs=[display_image, track_flag, rocket_session], show_progress=True)

            display_image.select(get_points_with_draw, inputs=[display_image, add_or_remove, rocket_session], outputs=[display_image, rocket_session])
            segment_type.select(set_segment_type, inputs=[segment_type, rocket_session], outputs=[rocket_session], show_progress=False)

        with gr.Row():
            with gr.Group():
                env_name = gr.Dropdown(env_list, multiselect=False, min_width=200, show_label=False, label="Env Name")
                world_seed = gr.Number(value=0, precision=0, label="World Seed")
                reset_btn = gr.Button("Reset Environment")
                reset_btn.click(
                    fn=reset_fn,
                    inputs=[env_name, world_seed, rocket_session],
                    outputs=[display_image, display_status, rocket_session],
                    show_progress=True,
                )

            with gr.Group():
                action_list = [x for x in NOOP_ACTION.keys()]
                # act_key = gr.Textbox("null", label="Action", show_label=False, min_width=200)
                act_key = gr.Dropdown(action_list, multiselect=False, min_width=200, show_label=False, label="Action")
                step_btn = gr.Button("Manually Step")
                step_btn.click(fn=step_fn, inputs=[act_key, rocket_session], outputs=[display_image, rocket_session], show_progress=False)

            with gr.Group():
                steps = gr.Slider(1, 600, 30, 1, label="Steps", show_label=False)
                play_btn = gr.Button("Call Rocket")
                play_btn.click(fn=loop_step_fn, inputs=[steps, rocket_session], outputs=[display_image, memory_length, display_status, rocket_session], show_progress=False)

            with gr.Group():
                staged_approach_steps = gr.Slider(0, 200, 20, 1, label="Approach Steps")
                staged_final_steps = gr.Slider(1, 200, 20, 1, label="Final Steps")
                staged_reset_memory = gr.Checkbox(True, label="Reset memory between stages")
                staged_play_btn = gr.Button("Call Rocket (2-Stage)")
                staged_play_btn.click(
                    fn=staged_loop_step_fn,
                    inputs=[staged_approach_steps, staged_final_steps, staged_reset_memory, rocket_session],
                    outputs=[display_image, memory_length, display_status, rocket_session],
                    show_progress=False,
                )

            with gr.Group():
                # memory_length = gr.Textbox(value="0", interactive=True)
                memory_length.render()
                clear_states_btn = gr.Button("Clear Memory")
                clear_states_btn.click(fn=clear_memory_fn, inputs=rocket_session, outputs=[display_image, memory_length, rocket_session], show_progress=False)
            
            make_video_btn = gr.Button("Make Video")
            save_video_btn = gr.DownloadButton("Download!!", visible=False)
            make_video_btn.click(make_video_fn, inputs=[rocket_session, make_video_btn, save_video_btn], outputs=[rocket_session, make_video_btn, save_video_btn], show_progress=False)
            save_video_btn.click(save_video_fn, inputs=[rocket_session, make_video_btn, save_video_btn], outputs=[rocket_session, make_video_btn, save_video_btn], show_progress=False)
        with gr.Row():
            with gr.Group():
                planner_task = gr.Textbox("collect wood from trees", label="Planner Task", show_label=True, min_width=200)
                planner_btn = gr.Button("Plan + Point")
                planner_output = gr.Textbox("", label="Planner Output", show_label=False, min_width=200)

            with gr.Group():
                molmo_text = gr.Textbox("pinpoint the", label="Molmo Text", show_label=True, min_width=200)
                molmo_btn = gr.Button("Generate")
                output_text = gr.Textbox("", label="Molmo Output", show_label=False, min_width=200)
                molmo_btn.click(molmo_fn, inputs=[molmo_text, molmo_session, rocket_session, display_image],outputs=[output_text, display_image],show_progress=False)

            with gr.Group():
                planned_steps = gr.Slider(1, 300, 40, 1, label="Planned Steps")
                replan_interval = gr.Slider(1, 100, 20, 1, label="Replan Interval")
                reset_memory_on_replan = gr.Checkbox(True, label="Reset memory on replan")
                planned_btn = gr.Button("Call Rocket (Planner Loop)")
                planned_btn.click(
                    planned_loop_step_fn,
                    inputs=[planner_task, planned_steps, replan_interval, reset_memory_on_replan, planner_session, molmo_session, rocket_session],
                    outputs=[display_image, memory_length, display_status, planner_output, molmo_text, segment_type, rocket_session],
                    show_progress=False,
                )
                planner_btn.click(
                    planner_point_fn,
                    inputs=[planner_task, planner_session, molmo_session, rocket_session, display_image],
                    outputs=[planner_output, molmo_text, segment_type, display_image, display_status],
                    show_progress=False,
                )

        demo.show_api = False
        demo.get_api_info = lambda all_endpoints=False: {"named_endpoints": {}, "unnamed_endpoints": {}}
        demo.queue(api_open=False)
        demo.launch(
            share=args.share,
            server_name=args.server_name,
            server_port=args.port,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--task-group", type=str, default="simple")
    parser.add_argument("--task-group-path", type=str, default="CraftJarvis/MineStudio_task_group.simple")
    parser.add_argument("--model-path", type=str, default="CraftJarvis/MineStudio_ROCKET-1.12w_EMA")
    parser.add_argument("--sam-path", type=str, required=True)
    parser.add_argument("--molmo-id", type=str, default="allenai/Molmo-7B-D-0924")
    parser.add_argument("--molmo-url", type=str, default="huggingface")
    parser.add_argument("--molmo-api-key", type=str, default="EMPTY")
    parser.add_argument("--planner-id", type=str, default="heuristic-planner")
    parser.add_argument("--planner-url", type=str, default="http://127.0.0.1:9164")
    parser.add_argument("--planner-api-key", type=str, default="EMPTY")
    args = parser.parse_args()
    
    draw_gradio_components(args)

if __name__ == "__main__":
    main()
