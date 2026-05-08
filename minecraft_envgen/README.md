# Minecraft EnvGen

This folder is a clean-room porting scaffold for moving EnvGen from Crafter to Minecraft without editing the original `EnvGen/` tree.

## Why split first

The current Crafter code mixes four different responsibilities in the same place:

1. LLM-facing curriculum format
2. Runtime environment API arguments
3. Feedback aggregation logic
4. Verification rules

That coupling is acceptable for a single backend, but it breaks down immediately for Minecraft:

- MineDojo and MineStudio expose different runtime knobs.
- Multimodal failure evidence is core research logic, not engine logic.
- Schema verification should validate engine APIs, while semantic verification should validate research intent.
- Train and eval reset policies should be explicit because fast reset is useful for training but unsafe for evaluation.

This folder separates those layers:

- `core/`: engine-agnostic curriculum draft, feedback packet, and prompt builder
- `verifiers/`: schema and semantic verifiers
- `adapters/`: backend-specific compilation for MineDojo and MineStudio
- `pipeline.py`: one place that runs verification, builds the prompt, and compiles backend runtime payloads

## Design choice

The MineDojo adapter intentionally validates only documented, common `minedojo.make(...)` kwargs and keeps unresolved curriculum interventions in `adapter_payload`.

That is deliberate. It prevents project-level intervention ideas from being falsely treated as official MineDojo API arguments. When you bind to a specific MineDojo task family later, you only extend the adapter, not the whole pipeline.

The MineStudio adapter goes further and maps some interventions directly into callback specs such as `FastResetCallback`, `HardResetCallback`, `CommandsCallback`, and `SummonMobsCallback`.

## Example

Run the demo pipeline:

```bash
python3 -m minecraft_envgen.scripts.demo_pipeline \
  --draft minecraft_envgen/examples/collect_diamond_draft.json \
  --feedback minecraft_envgen/examples/collect_diamond_feedback.json \
  --adapter all \
  --output-dir /tmp/minecraft_envgen_demo
```

This produces:

- a prompt that can be sent to an LLM
- a verification report
- compiled MineDojo runtime JSON
- compiled MineStudio runtime JSON

## Run the MineStudio VPT baseline with EnvGen

Once you have a curriculum draft, you can run it directly against MineStudio's VPT baseline and emit a post-hoc feedback packet with multimodal failure evidence:

```bash
conda run -n minestudio python -m minecraft_envgen.scripts.run_vpt_envgen \
  --draft minecraft_envgen/examples/collect_diamond_draft.json \
  --split eval \
  --episodes 2 \
  --steps 600 \
  --output-dir /tmp/minecraft_envgen_vpt
```

This writes:

- `prompt.txt`, `report.json`, and `compiled.json`
- MineStudio rollout artifacts under `episodes/`
- `summary.json` with the success rate
- `feedback_packet.json` with scalar metrics and multimodal failure evidence

## Run VPT-RL ablations

To compare the four experiment conditions directly, use the ablation runner:

```bash
conda run -n minestudio python -m minecraft_envgen.scripts.run_vpt_ablation \
  --mode baseline \
  --draft minecraft_envgen/examples/collect_wood_draft.json \
  --split eval \
  --cycles 1 \
  --episodes 10 \
  --steps 600 \
  --output-dir /tmp/minecraft_envgen_ablation_baseline
```

Supported modes:

- `baseline`: VPT-RL only on the fixed compiled curriculum environment. This is the fair control for comparing against the EnvGen modes.
- `baseline_default`: VPT-RL only on the default MineStudio sandbox environment
- `scalar_envgen`: VPT-RL + scalar-only EnvGen prompting
- `multimodal_envgen`: VPT-RL + multimodal EnvGen prompting
- `multimodal_envgen_verifier`: VPT-RL + multimodal EnvGen prompting + verifier gating

For the three EnvGen modes, set `--cycles` to a value greater than `1` and provide an OpenAI key via `OPENAI_API_KEY`, `API_KEY`, or `--dotenv path/to/.env`.

The current MineStudio runtime bridge supports:

- callback instantiation from compiled specs
- VPT baseline evaluation with `CraftJarvis/MineStudio_VPT.rl_from_early_game_2x`
- post-hoc evidence packing from saved video / info / action artifacts
- voxel evidence when the draft requests `privileged_observations=["voxel"]`

## Next extension points

- Replace the example prompt builder with your actual LLM caller.
- Add task-family-specific MineDojo bindings inside `adapters/minedojo.py`.
- Add richer MineStudio reward, recorder, and benchmark callbacks.
- Expand the semantic verifier with your skill ontology and intervention taxonomy.
