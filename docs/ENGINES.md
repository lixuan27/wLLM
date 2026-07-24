# External engine bindings

wLLM's own runtime (serving + native) is self-contained. Some *apps*
additionally delegate stages to external engines installed on the host.
Those engines are **never vendored and never imported by name** in this
tree — each is resolved at runtime through an environment variable, so
the same app code runs against whichever compatible engine build a site
provides, and the published tree stays engine-neutral.

## Omni-modal stage engine

Used by the `qwen3_omni` and `liveavatar` apps for BLACK_BOX stages
(async multi-stage serving with an `AsyncOmni` entrypoint, stage-config
YAMLs, paged attention, codec predictors).

| Variable | Meaning |
|---|---|
| `WLLM_OMNI_ENGINE` | Package name of the installed omni-modal serving engine. Must expose `AsyncOmni` at the top level and the stage/scheduler layout referenced by the app stage configs. |

Binding module: `wllm/engines/omni.py`. Stage-config YAMLs reference
engine-internal classes through the `__WLLM_OMNI_ENGINE__` placeholder;
`render_stage_config()` substitutes the bound package name right before
the config path is handed to the engine.

**In-tree default**: `WLLM_OMNI_ENGINE=wllm.omni` binds the repository's
own staged engine (`wllm/omni/`), which implements the same contract —
`AsyncOmni`, the stage-config schema (it also resolves the placeholder
natively), AR/generation stage schedulers, and app-compatible output
objects. Real model runners register via `wllm.omni.stages.register_stage`;
unregistered models fail closed rather than silently degrading.

If a stage needs the engine and the variable is unset, the app fails
closed at import with `OmniEngineNotBound` — it never silently falls
back to a slower path.

## Model-substrate catalog (L0 opaque launches)

`wllm/backends/catalog/` imports model manifests from a locally checked
out model-substrate workspace and launches its models as opaque (L0)
subprocesses in their own conda envs.

| Variable | Meaning |
|---|---|
| `WLLM_SUBSTRATE_ROOT` | Path to the substrate checkout (manifest tree is auto-discovered at `<pkg>/data/models/catalog`). |
| `WLLM_SUBSTRATE_JOB_MODULE` | Dotted module exposing the substrate's `infer` job CLI (`python -m <module> infer ...`). |
| `WLLM_SUBSTRATE_CKPT_ENV` | Name of the env var the substrate reads for its checkpoint directory (optional). |
| `WLLM_SUBSTRATE_UNIFIED_ENV` | Conda env name for manifests that declare the `unified` environment. |

Launching without `WLLM_SUBSTRATE_JOB_MODULE` fails closed with a
pointer to this document.

## AR text engine

Plain `vllm` (the open-source AR serving engine) is consumed as an
ordinary installed dependency by some app stages; no binding indirection
is applied to it.

## Site configuration

Put the bindings in an untracked env file and source it from your job
scripts, e.g.:

```bash
# .wllm.env (untracked)
export WLLM_OMNI_ENGINE=<your omni engine package>
export WLLM_SUBSTRATE_ROOT=$HOME/substrate-checkout
export WLLM_SUBSTRATE_JOB_MODULE=<pkg>.studio.workspace_job
export WLLM_SUBSTRATE_CKPT_ENV=<PKG>_CKPT_DIR
export WLLM_SUBSTRATE_UNIFIED_ENV=<pkg>-unified-cu128
```
