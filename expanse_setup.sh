#!/bin/bash
# expanse_setup.sh - sets up the Spark environment on SDSC Expanse
# for the EB-NeRD project (DSC 232R)
#
# Usage:
#   chmod +x expanse_setup.sh
#   bash expanse_setup.sh

set -euo pipefail

echo "Setting up EB-NeRD Spark project on Expanse..."

# Create project directories
PROJECT_DIR="${HOME}/dSalcidoSparkProject"
DATA_DIR="${PROJECT_DIR}/data"
mkdir -p "${DATA_DIR}"
echo "[1/5] Project directory: ${PROJECT_DIR}"

# Load modules
echo "[2/5] Loading modules..."
module purge
module load cpu
module load spark/3.5.0
module load anaconda3
echo "       Spark home: ${SPARK_HOME:-not set}"

# Create conda env if it doesn't exist
ENV_NAME="ebnerd_spark"
if conda info --envs | grep -q "${ENV_NAME}"; then
    echo "[3/5] Conda env '${ENV_NAME}' already exists, skipping"
else
    echo "[3/5] Creating conda env '${ENV_NAME}'..."
    conda create -y -n "${ENV_NAME}" python=3.10
fi

source activate "${ENV_NAME}" 2>/dev/null || conda activate "${ENV_NAME}"

# Install packages
echo "[4/5] Installing Python packages..."
pip install --quiet --upgrade pip
pip install --quiet pyspark matplotlib ipykernel

python -m ipykernel install --user --name "${ENV_NAME}" \
    --display-name "PySpark (${ENV_NAME})"

# Done
echo ""
echo "Setup complete."
echo "  Project dir : ${PROJECT_DIR}"
echo "  Data dir    : ${DATA_DIR}"
echo "  Conda env   : ${ENV_NAME}"
echo "  Python      : $(python --version 2>&1)"
echo "  Spark       : $(pyspark --version 2>&1 | head -1 || echo 'check module')"
echo ""
echo "Next steps:"
echo "  1. Download the data:"
echo "     cd ~/ebnerd_data"
echo ""
echo "     wget -O ebnerd_large.zip 'https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip'"
echo "     unzip ebnerd_large.zip -d ebnerd_large"
echo ""
echo "     wget -O contrastive_vector.zip 'https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_contrastive_vector.zip'"
echo "     unzip contrastive_vector.zip -d artifacts"
echo ""
echo "     wget -O xlm_roberta.zip 'https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/FacebookAI_xlm_roberta_base.zip'"
echo "     unzip xlm_roberta.zip -d artifacts"
echo ""
echo "  2. Open Jupyter via the Expanse portal or SLURM"
echo "  3. Open MS2_Data_Exploration.ipynb and set DATA_ROOT to ~/ebnerd_data"
