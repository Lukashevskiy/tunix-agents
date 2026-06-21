#!/usr/bin/env python3
"""Write real CrafText/Caged eager-rollout throughput evidence as dashboard JSON."""
from __future__ import annotations
import argparse, json, platform, subprocess, time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import jax
import jax.numpy as jnp
from tunix_craftext.config import load_mvp_config
from tunix_craftext.random_policy import sample_masked_actions
from tunix_craftext.runtime import build_craftext_runtime

def point(path: Path, batch: int, horizon: int, repeats: int) -> dict[str, object]:
    cfg = load_mvp_config(path); cfg = replace(cfg, environment=replace(cfg.environment, batch_size=batch, horizon=horizon))
    runtime = build_craftext_runtime(cfg)
    def rollout() -> jax.Array:
        reset = jax.vmap(runtime.adapter.reset)(jax.random.split(jax.random.PRNGKey(cfg.run.seed), batch))
        state, mask = reset.state, jnp.broadcast_to(reset.action_mask, (batch, runtime.action_count))
        keys = jax.random.split(jax.random.PRNGKey(101), horizon * batch * 2).reshape(horizon, 2, batch, 2)
        def scan_step(carry, step_keys):
            current_state, current_mask = carry
            action = sample_masked_actions(step_keys[0], current_mask)
            step = jax.vmap(runtime.adapter.step)(step_keys[1], current_state, action)
            return (step.state, step.action_mask), step.reward
        _, rewards = jax.lax.scan(scan_step, (state, mask), keys)
        return rewards
    compiled = jax.jit(rollout)
    start=time.perf_counter(); jax.block_until_ready(compiled()); compile_ms=(time.perf_counter()-start)*1000
    times=[]
    for _ in range(repeats):
        start=time.perf_counter(); jax.block_until_ready(compiled()); times.append((time.perf_counter()-start)*1000)
    steady=sum(times)/len(times)
    return {"variant":cfg.run.name,"config":str(path),"batch_size":batch,"horizon":horizon,"compile_ms":round(compile_ms,3),"steady_state_ms":round(steady,3),"env_steps_per_second":round(batch*horizon/(steady/1000),3)}

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--configs",nargs="+",type=Path,required=True); p.add_argument("--batch-sizes",nargs="+",type=int,default=[1,2,8,32]); p.add_argument("--horizons",nargs="+",type=int,default=[8,32,128,512]); p.add_argument("--repeats",type=int,default=20); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    points=[]
    payload={"schema":"tunix-craftext.environment-benchmark/v1","timestamp":datetime.now(timezone.utc).isoformat(),"commit":subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip(),"hardware":f"{platform.system()} {platform.machine()} / {jax.default_backend()}","points":points}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for config in a.configs:
        for batch in a.batch_sizes:
            for horizon in a.horizons:
                try: points.append(point(config,batch,horizon,a.repeats))
                except Exception as error: points.append({"variant":config.stem,"config":str(config),"batch_size":batch,"horizon":horizon,"status":"failed","error":str(error)})
                temporary=a.output.with_suffix(".tmp"); temporary.write_text(json.dumps(payload,indent=2),encoding="utf-8"); temporary.replace(a.output)
if __name__ == "__main__": main()
