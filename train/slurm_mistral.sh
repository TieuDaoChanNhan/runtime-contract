#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=05:00:00
#SBATCH --output=out/slurm-%j.log
#SBATCH --error=out/slurm-%j.err
set -euo pipefail

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Isolated venv - the repo's own .venv can't have both the train and
# inference extras installed at once (pyproject.toml [tool.uv] conflicts).
source "${MISTRAL_VENV:?set MISTRAL_VENV to your Mistral training venv}/bin/activate"

# Wandb offline mode disabled — GPU nodes have no internet.
# Training metrics are parsed from slurm logs instead.
# export WANDB_MODE=offline

mkdir -p out
exec "$@"
