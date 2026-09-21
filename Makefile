# =============================================================================
# Configuration Variables
# =============================================================================
PYTHON := uv run python
DATA_DIR := data
CONFIG_DIR := codeact_runtime/task_configs
MODEL_NAME := gemini-3-flash
TRAIN_TRACES_TO_GENERATE := 5
TEST_TASKS = 100
MAX_TRAIN_TASKS = 1000

# Derived Paths
TASKS_DIR := $(DATA_DIR)/tasks
TRACES_DIR := $(DATA_DIR)/traces/knapsack/$(MODEL_NAME)
TRAINING_DATA_DIR := $(DATA_DIR)/training/$(MODEL_NAME)

# LoRA paths (override with your own trained adapter output dir, e.g. make
# serve-persistent-lora PERSISTENT_LORA=out/qwen3-8b-persistent-<your-run>)
PERSISTENT_LORA := out/qwen3-8b-persistent-lora
STATELESS_LORA  := out/qwen3-8b-stateless-lora

# Paired data paths
PAIRED_DATA_DIR        := paired/train/out/paired_data
PAIRED_PERSISTENT_DATA := $(PAIRED_DATA_DIR)/persistent/traces.jsonl
PAIRED_STATELESS_DATA  := $(PAIRED_DATA_DIR)/stateless/traces.jsonl

# SLURM cluster config (SLURM_PARTITION and SLURM_ACCOUNT come from .env)
SBATCH := sbatch --partition=$(SLURM_PARTITION) --account=$(SLURM_ACCOUNT)
HF_CACHE ?= $(HF_HOME)

# =============================================================================
# Phony Targets
# =============================================================================
.PHONY: lint-all format-notebooks format-check format pyright ruff ruff-fix sort-imports check
.PHONY: serve serve-lora serve-persistent-lora serve-stateless-lora
.PHONY: train-persistent train-stateless help
.PHONY: slurm-train-setup slurm-train-persistent slurm-train-stateless slurm-status
.PHONY: slurm-serve slurm-serve-setup slurm-serve-persistent-lora slurm-serve-stateless-lora
.PHONY: $(foreach r,persistent stateless,$(foreach m,base persistent stateless,$(foreach d,easy medium,slurm-bench-$(r)-$(m)-$(d))))
.PHONY: tasks split traces-persistent traces-stateless stats prepare-data
.PHONY: traces-persistent-rule traces-stateless-rule traces-persistent-nav traces-stateless-nav
.PHONY: gen-benchmark-configs
.PHONY: $(foreach r,persistent stateless,$(foreach m,base persistent stateless,$(foreach d,easy medium,bench-$(r)-$(m)-$(d))))

help:
	@echo "CodeAct-Runtime Commands (/nothink enabled)"
	@echo ""
	@echo "Data Pipeline (End-to-End):"
	@echo "  make tasks                        Generate all tasks (easy/hard)"
	@echo "  make split                        Split tasks into train/test sets"
	@echo "  make traces-persistent            Generate persistent teacher traces (knapsack)"
	@echo "  make traces-stateless             Generate stateless teacher traces (knapsack)"
	@echo "  make traces-persistent-rule       Generate persistent teacher traces (rule_diagnosis)"
	@echo "  make traces-stateless-rule        Generate stateless teacher traces (rule_diagnosis)"
	@echo "  make traces-persistent-nav        Generate persistent teacher traces (navigation)"
	@echo "  make traces-stateless-nav         Generate stateless teacher traces (navigation)"
	@echo "  make prepare-data                 Format traces for Axolotl training"
	@echo ""
	@echo "Inference (local):"
	@echo "  make serve                        Base model (bf16, 40k ctx)"
	@echo "  make serve-lora LORA=<path>       With LoRA adapter"
	@echo "  make serve-persistent-lora        Persistent state LoRA"
	@echo "  make serve-stateless-lora         Stateless LoRA"
	@echo ""
	@echo "Inference (SLURM):"
	@echo "  make slurm-serve                  Base model via SLURM"
	@echo "  make slurm-serve-persistent-lora  Persistent LoRA via SLURM"
	@echo "  make slurm-serve-stateless-lora   Stateless LoRA via SLURM"
	@echo ""
	@echo "Training (assistant-only loss):"
	@echo "  make train-persistent             CodeAct persistent LoRA"
	@echo "  make train-stateless              CodeAct stateless LoRA"
	@echo ""
	@echo "Config Generation:"
	@echo "  make gen-benchmark-configs        Regenerate benchmark + teacher configs from template"
	@echo ""
	@echo "Benchmarks (bench-{runtime}-{model}-{difficulty}):"
	@echo "  make bench-persistent-base-easy   Persistent runtime, base model, easy tasks"
	@echo "  ... 12 total: {persistent,stateless} x {base,persistent,stateless} x {easy,hard}"
	@echo ""
	@echo "Benchmarks (SLURM — serve + bench in one job):"
	@echo "  make slurm-bench-persistent-base-easy"
	@echo "  ... 12 total: slurm-bench-{runtime}-{model}-{difficulty}"
	@echo ""
	@echo "Linting:"
	@echo "  make lint-all                     Run all linters with fixes"
	@echo "  make check                        Run all checks (no fixes)"

# =============================================================================
# Data Pipeline (Reproduction)
# =============================================================================

tasks:
	@echo "--- Generating Tasks ---"
	$(PYTHON) -m codeact_runtime.cli --config $(CONFIG_DIR)/easy.json --out $(TASKS_DIR)/easy
	$(PYTHON) -m codeact_runtime.cli --config $(CONFIG_DIR)/medium.json --out $(TASKS_DIR)/medium
	$(PYTHON) -m codeact_runtime.cli --config $(CONFIG_DIR)/hard.json --out $(TASKS_DIR)/hard

split:
	@echo "--- Splitting Dataset ---"
	$(PYTHON) scripts/dataset_split.py --dir $(TASKS_DIR)/easy/tasks --test-count $(TEST_TASKS)
	$(PYTHON) scripts/dataset_split.py --dir $(TASKS_DIR)/medium/tasks --test-count $(TEST_TASKS)
	$(PYTHON) scripts/dataset_split.py --dir $(TASKS_DIR)/hard/tasks --test-count $(TEST_TASKS)

traces-persistent:
	@echo "--- Generating Persistent Traces ---"
	$(PYTHON) codeact_runtime/benchmark/benchmark.py \
		--config configs/generate_traces/knapsack_persistent.yaml \
		--tasks_root $(TASKS_DIR)/easy \
		--split train \
		--max-examples $(TRAIN_TRACES_TO_GENERATE)

traces-persistent-rule:
	@echo "--- Generating Persistent Traces ---"
	$(PYTHON) codeact_runtime/benchmark/benchmark.py \
		--config configs/generate_traces/rule_diagnosis_persistent.yaml \
		--tasks_root $(TASKS_DIR)/easy \
		--split train \
		--max-examples $(TRAIN_TRACES_TO_GENERATE)

traces-stateless-rule:
	@echo "--- Generating Stateless Traces ---"
	$(PYTHON) codeact_runtime/benchmark/benchmark.py \
		--config configs/generate_traces/rule_diagnosis_stateless.yaml \
		--tasks_root $(TASKS_DIR)/easy \
		--split train \
		--max-examples $(TRAIN_TRACES_TO_GENERATE)

traces-persistent-nav:
	@echo "--- Generating Persistent Traces ---"
	$(PYTHON) codeact_runtime/benchmark/benchmark.py \
		--config configs/generate_traces/navigation_persistent.yaml \
		--tasks_root $(TASKS_DIR)/easy \
		--split train \
		--max-examples $(TRAIN_TRACES_TO_GENERATE)

traces-stateless-nav:
	@echo "--- Generating Stateless Traces ---"
	$(PYTHON) codeact_runtime/benchmark/benchmark.py \
		--config configs/generate_traces/navigation_stateless.yaml \
		--tasks_root $(TASKS_DIR)/easy \
		--split train \
		--max-examples $(TRAIN_TRACES_TO_GENERATE)

# 4. Generate Traces (Stateless/Reset)
traces-stateless:
	@echo "--- Generating Stateless Traces ---"
	$(PYTHON) codeact_runtime/benchmark/benchmark.py \
		--config configs/generate_traces/knapsack_stateless.yaml \
		--tasks_root $(TASKS_DIR)/easy \
		--split train \
		--max-examples $(TRAIN_TRACES_TO_GENERATE)

stats:
	@echo "--- Calculating Trace Statistics ---"
	$(PYTHON) codeact_runtime/get_trace_stats.py \
		$(TRACES_DIR)/persistent_state/results/persistent-$(MODEL_NAME)/knapsack/ \
		$(TRACES_DIR)/stateless/results/stateless-$(MODEL_NAME)/knapsack/

prepare-data:
	@echo "--- Converting Traces to Axolotl Format ---"
	$(PYTHON) train/prepare_data.py \
		--trace-root $(TRACES_DIR)/persistent_state \
		--output-dir $(TRAINING_DATA_DIR)/persistent_state \
		--max-samples $(MAX_TRAIN_TASKS)

prepare-data-stateless:
	@echo "--- Converting Traces to Axolotl Format ---"
	$(PYTHON) train/prepare_data.py \
		--trace-root $(TRACES_DIR)/stateless \
		--output-dir $(TRAINING_DATA_DIR)/stateless \
		--max-samples $(MAX_TRAIN_TASKS)

# =============================================================================
# Inference (vLLM) — bf16, 40k context, /nothink
# =============================================================================

serve:
	NOTHINK=true ./train/inference.sh

serve-lora:
	NOTHINK=true ./train/inference.sh --lora $(LORA)

serve-persistent-lora:
	NOTHINK=true LORA_NAME=qwen3-8b-persistent-lora ./train/inference.sh --lora $(PERSISTENT_LORA)

serve-stateless-lora:
	NOTHINK=true LORA_NAME=qwen3-8b-stateless-lora ./train/inference.sh --lora $(STATELESS_LORA)

# =============================================================================
# Training — QLoRA, 16k context, 3 epochs, assistant-only loss
# =============================================================================

train-persistent:
	./train/train.sh \
		--base-config train/configs/axolotl_qwen3_8b_persistent.yaml \
		--output-dir out/qwen3-8b-persistent \
		--session train-persistent

train-stateless:
	./train/train.sh \
		--base-config train/configs/axolotl_qwen3_8b_stateless.yaml \
		--output-dir out/qwen3-8b-stateless \
		--session train-stateless


# =============================================================================
# SLURM Training (anonymized cluster, GH200 nodes)
# =============================================================================

WANDB_PROJECT ?= codeact_runtime

slurm-train-setup:
	rm -rf .venv
	uv sync --extra train --extra fa

TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

slurm-train-persistent: slurm-train-setup
	$(SBATCH) --job-name=anon-persistent train/slurm.sh \
		./train/train.sh \
		--data-file $(PAIRED_PERSISTENT_DATA) \
		--base-config train/configs/axolotl_qwen3_8b_persistent.yaml \
		--output-dir out/qwen3-8b-persistent-$(TIMESTAMP) \
		--wandb-project $(WANDB_PROJECT) \
		--session train-persistent

slurm-train-stateless: slurm-train-setup
	$(SBATCH) --job-name=anon-stateless train/slurm.sh \
		./train/train.sh \
		--data-file $(PAIRED_STATELESS_DATA) \
		--base-config train/configs/axolotl_qwen3_8b_stateless.yaml \
		--output-dir out/qwen3-8b-stateless-$(TIMESTAMP) \
		--wandb-project $(WANDB_PROJECT) \
		--session train-stateless

slurm-status:
	@squeue -u $$USER --format="%.10i %.15j %.8T %.10M %.6D %.4C %.20R" 2>/dev/null || echo "No jobs found"
	@echo ""
	@ls -t out/slurm-*.log 2>/dev/null | head -1 | xargs -I{} sh -c 'echo "Latest log: {}"; echo "---"; tail -5 "{}"' 2>/dev/null || true

# =============================================================================
# SLURM Inference (anonymized cluster, GH200 nodes)
# =============================================================================

slurm-serve-setup:
	rm -rf .venv
	uv sync --extra inference

slurm-serve: slurm-serve-setup
	$(SBATCH) --job-name=anon-serve --time=16:00:00 train/slurm.sh \
		env NOTHINK=true ./train/inference.sh --foreground

slurm-serve-persistent-lora: slurm-serve-setup
	$(SBATCH) --job-name=anon-serve-plora --time=16:00:00 train/slurm.sh \
		env NOTHINK=true LORA_NAME=qwen3-8b-persistent-lora ./train/inference.sh --foreground --lora $(PERSISTENT_LORA)

slurm-serve-stateless-lora: slurm-serve-setup
	$(SBATCH) --job-name=anon-serve-slora --time=16:00:00 train/slurm.sh \
		env NOTHINK=true LORA_NAME=qwen3-8b-stateless-lora ./train/inference.sh --foreground --lora $(STATELESS_LORA)

# =============================================================================
# SLURM Benchmarks (serve + bench in one job)
# =============================================================================

SLURM_BENCH_TIME ?= 12:00:00

# Map model name to --lora/--lora-name args for slurm_bench.sh
_lora_args_base :=
_lora_args_persistent := --lora $(PERSISTENT_LORA) --lora-name qwen3-8b-persistent-lora
_lora_args_stateless  := --lora $(STATELESS_LORA) --lora-name qwen3-8b-stateless-lora

define slurm-bench-target
slurm-bench-$(1)-$(2)-$(3): slurm-serve-setup
	$(SBATCH) --job-name=anon-bench-$(1)-$(2)-$(3) --time=$(SLURM_BENCH_TIME) train/slurm.sh \
		./train/slurm_bench.sh \
		$(_lora_args_$(2)) \
		--config train/assets/benchmark_configs/$(1)_$(2)_$(3).yaml \
		--tasks-root $$(TASKS_DIR)/$(3)/test
endef

$(foreach r,persistent stateless, \
  $(foreach m,base persistent stateless, \
    $(foreach d,easy medium, \
      $(eval $(call slurm-bench-target,$(r),$(m),$(d))))))

# =============================================================================
# Config Generation
# =============================================================================

gen-benchmark-configs:
	$(PYTHON) train/assets/benchmark_configs/_generate.py

# =============================================================================
# Benchmarks — bench-{runtime}-{model}-{difficulty}
# 2 runtimes x 3 models x 2 difficulties = 12 targets
# =============================================================================

define bench-target
bench-$(1)-$(2)-$(3):
	./train/benchmark.sh --config train/assets/benchmark_configs/$(1)_$(2)_$(3).yaml \
		--tasks-root $$(TASKS_DIR)/$(3)/test --session bench-$(1)-$(2)-$(3)
endef

$(foreach r,persistent stateless, \
  $(foreach m,base persistent stateless, \
    $(foreach d,easy medium, \
      $(eval $(call bench-target,$(r),$(m),$(d))))))

# =============================================================================
# Linting
# =============================================================================

lint-all: format ruff-fix sort-imports pyright

format-notebooks:
	uv run nbqa isort . && uv run nbqa ruff . --fix

format-check:
	uv run ruff format --check

format:
	uv run ruff format

pyright:
	uv run pyright

ruff:
	uv run ruff check .

ruff-fix:
	uv run ruff check . --fix

sort-imports:
	uv run ruff check --select I --fix

check: format-check ruff pyright

# Regenerate paper numbers JSON and render main.tex/appendix.tex from Jinja templates.
paper-numbers:
	uv run python -m scripts.build_paper_numbers
figures: paper-numbers
	uv run python scripts/plot_figures.py
