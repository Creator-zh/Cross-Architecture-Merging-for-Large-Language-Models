from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ACITaskPreset:
    name: str
    target_hf_id: str
    target_local_dir: str
    reference_hf_id: str
    reference_local_dir: str
    source_hf_id: str
    source_local_dir: str
    beta: float
    eval_kind: str
    primary_eval_tasks: tuple[str, ...]
    extended_eval_tasks: tuple[str, ...] = ()
    sft_dataset_type: str = ""
    sft_learning_rate: float = 0.0
    sft_samples: int = 2000


SHARED_SOURCE_ID = "unsloth/Llama-3.1-8B-Instruct"
SHARED_SOURCE_DIR = "Llama-3.1-8B-Instruct"
REFERENCE_BASE_ID = "unsloth/Llama-3.2-1B"
REFERENCE_BASE_DIR = "Llama-3.2-1B"
REFERENCE_INSTRUCT_ID = "unsloth/Llama-3.2-1B-Instruct"
REFERENCE_INSTRUCT_DIR = "Llama-3.2-1B-Instruct"


TASK_PRESETS: dict[str, ACITaskPreset] = {
    "medical": ACITaskPreset(
        name="medical",
        target_hf_id="PathFinderKR/Llama-3-1B-Medical-Instruct",
        target_local_dir="llama3-1b-med",
        reference_hf_id=REFERENCE_BASE_ID,
        reference_local_dir=REFERENCE_BASE_DIR,
        source_hf_id=SHARED_SOURCE_ID,
        source_local_dir=SHARED_SOURCE_DIR,
        beta=0.03,
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
    "thai": ACITaskPreset(
        name="thai",
        target_hf_id="typhoon-ai/llama3.2-typhoon2-1b-instruct",
        target_local_dir="llama3.2-typhoon2-1b-instruct",
        reference_hf_id=REFERENCE_INSTRUCT_ID,
        reference_local_dir=REFERENCE_INSTRUCT_DIR,
        source_hf_id=SHARED_SOURCE_ID,
        source_local_dir=SHARED_SOURCE_DIR,
        beta=0.01,
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
        sft_samples=8000,
    ),
    "malay": ACITaskPreset(
        name="malay",
        target_hf_id="mesolitica/Malaysian-Llama-3.2-1B-Instruct",
        target_local_dir="Malaysian-Llama-3.2-1B-Instruct-v0.1",
        reference_hf_id=REFERENCE_INSTRUCT_ID,
        reference_local_dir=REFERENCE_INSTRUCT_DIR,
        source_hf_id=SHARED_SOURCE_ID,
        source_local_dir=SHARED_SOURCE_DIR,
        beta=0.10,
        eval_kind="malay_mmlu",
        primary_eval_tasks=("MalayMMLU",),
        sft_dataset_type="malaysian_sft",
        sft_learning_rate=1e-6,
    ),
}


def get_task_preset(name: str) -> ACITaskPreset:
    try:
        return TASK_PRESETS[name]
    except KeyError as error:
        raise ValueError(f"Unknown ACI task preset: {name}") from error


def aci_run_name(task: str, beta: float | None = None) -> str:
    resolved = get_task_preset(task).beta if beta is None else float(beta)
    return f"{task}_aci_beta{resolved:g}"


def aci_sft_run_name(task: str, beta: float | None = None) -> str:
    return f"{aci_run_name(task, beta)}_sft"
