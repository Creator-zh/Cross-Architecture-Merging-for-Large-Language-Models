from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DFOPTaskPreset:
    name: str
    target_hf_id: str
    target_local_dir: str
    source_hf_id: str
    source_local_dir: str
    matched_beta: float
    eval_kind: str
    primary_eval_tasks: tuple[str, ...]
    extended_eval_tasks: tuple[str, ...] = ()
    sft_dataset_type: str = ""
    sft_learning_rate: float = 0.0
    sft_samples_declared: int = 2000
    sft_samples_legacy: int = 2000


SHARED_LLAMA_8B_ID = "unsloth/Llama-3.1-8B-Instruct"
SHARED_LLAMA_8B_DIR = "Llama-3.1-8B-Instruct"


TASK_PRESETS: dict[str, DFOPTaskPreset] = {
    "medical": DFOPTaskPreset(
        name="medical",
        target_hf_id="PathFinderKR/Llama-3-1B-Medical-Instruct",
        target_local_dir="llama3-1b-med",
        source_hf_id=SHARED_LLAMA_8B_ID,
        source_local_dir=SHARED_LLAMA_8B_DIR,
        matched_beta=0.03,
        eval_kind="lm_eval",
        primary_eval_tasks=(
            "medqa_4options",
            "mmlu_anatomy",
            "medmcqa",
            "mmlu_clinical_knowledge",
            "mmlu_college_biology",
            "mmlu_college_medicine",
            "mmlu_medical_genetics",
            "mmlu_professional_medicine",
        ),
        sft_dataset_type="medical_llama3",
        sft_learning_rate=3e-7,
    ),
    "thai": DFOPTaskPreset(
        name="thai",
        target_hf_id="scb10x/llama3.2-typhoon2-1b-instruct",
        target_local_dir="llama3.2-typhoon2-1b-instruct",
        source_hf_id=SHARED_LLAMA_8B_ID,
        source_local_dir=SHARED_LLAMA_8B_DIR,
        matched_beta=0.01,
        eval_kind="lm_eval",
        primary_eval_tasks=("xcopa_th", "xquad_th", "xnli_th"),
        extended_eval_tasks=(
            "belebele_tha_Thai",
            "mgsm_direct_th",
            "mgsm_native_cot_th",
            "mmlu_prox_lite_th_other",
        ),
        sft_dataset_type="fineweb_thai",
        sft_learning_rate=1e-7,
        sft_samples_declared=8000,
        sft_samples_legacy=2000,
    ),
    "malay": DFOPTaskPreset(
        name="malay",
        target_hf_id="mesolitica/Malaysian-Llama-3.2-1B-Instruct",
        target_local_dir="Malaysian-Llama-3.2-1B-Instruct-v0.1",
        source_hf_id=SHARED_LLAMA_8B_ID,
        source_local_dir=SHARED_LLAMA_8B_DIR,
        matched_beta=0.10,
        eval_kind="malay_mmlu",
        primary_eval_tasks=("MalayMMLU",),
        sft_dataset_type="malaysian_sft",
        sft_learning_rate=1e-6,
    ),
}


def get_task_preset(name: str) -> DFOPTaskPreset:
    try:
        return TASK_PRESETS[name]
    except KeyError as error:
        raise ValueError(f"Unknown DFOP task preset: {name}") from error


def fusion_beta(track: str, task: str) -> float:
    """Universal track uses 0.05; matched track uses each paper task alpha."""
    if track == "universal":
        return 0.05
    return float(get_task_preset(task).matched_beta)


def dfop_fusion_run_name(
    task: str,
    mode: str,
    track: str,
    rank: int,
    top_source_layers: int,
    beta: float | None = None,
    *,
    route_solver: str = "row_softmax_topk",
    route_grouping: str = "independent",
) -> str:
    """Directory name for a fused DFOP run without route-config collisions."""
    resolved_beta = fusion_beta(track, task) if beta is None else float(beta)
    route_label = (
        f"top{int(top_source_layers)}"
        if route_solver == "row_softmax_topk"
        else "balanced"
    )
    grouping_label = "" if route_grouping == "independent" else f"_{route_grouping}"
    return (
        f"{task}_{mode}_{track}_r{int(rank)}_{route_label}{grouping_label}"
        f"_beta{resolved_beta:g}"
    )


def dfop_sft_run_name(
    task: str,
    mode: str,
    track: str,
    rank: int,
    top_source_layers: int,
    *,
    route_solver: str = "row_softmax_topk",
    route_grouping: str = "independent",
) -> str:
    """Directory name for a post-merge SFT run."""
    route_label = (
        f"top{int(top_source_layers)}"
        if route_solver == "row_softmax_topk"
        else "balanced"
    )
    grouping_label = "" if route_grouping == "independent" else f"_{route_grouping}"
    return f"{task}_{mode}_{track}_r{int(rank)}_{route_label}{grouping_label}"
