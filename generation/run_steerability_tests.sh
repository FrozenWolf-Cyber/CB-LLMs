#!/bin/bash
# Run steerability tests on saved wandb runs with multiple classifier weights
# Hardcoded for all 4 datasets

set -e

RUN_IDS_PICKLE=${1:-"run_ids.pkl"}
WANDB_PROJECT=${2:-"cbm-generation-new"}
WANDB_ENTITY=${3:-""}

CLASSIFIER_SUFFIXES="_seed42,_seed123,_seed456"

# Build entity flag
ENTITY_FLAG=""
if [ -n "$WANDB_ENTITY" ]; then
    ENTITY_FLAG="--wandb_entity $WANDB_ENTITY"
fi

# ===================== SetFit/sst2 =====================
echo "=========================================="
echo "Running steerability tests for SetFit/sst2"
echo "=========================================="
python resume_steerability_test.py \
    --run_ids_pickle "$RUN_IDS_PICKLE" \
    --dataset "SetFit/sst2" \
    --classifier_weight_suffixes "$CLASSIFIER_SUFFIXES" \
    --wandb_project "$WANDB_PROJECT" \
    $ENTITY_FLAG

# ===================== ag_news =====================
echo "=========================================="
echo "Running steerability tests for ag_news"
echo "=========================================="
python resume_steerability_test.py \
    --run_ids_pickle "$RUN_IDS_PICKLE" \
    --dataset "ag_news" \
    --classifier_weight_suffixes "$CLASSIFIER_SUFFIXES" \
    --wandb_project "$WANDB_PROJECT" \
    $ENTITY_FLAG

# ===================== yelp_polarity =====================
echo "=========================================="
echo "Running steerability tests for yelp_polarity"
echo "=========================================="
python resume_steerability_test.py \
    --run_ids_pickle "$RUN_IDS_PICKLE" \
    --dataset "yelp_polarity" \
    --classifier_weight_suffixes "$CLASSIFIER_SUFFIXES" \
    --wandb_project "$WANDB_PROJECT" \
    $ENTITY_FLAG

# ===================== dbpedia_14 =====================
echo "=========================================="
echo "Running steerability tests for dbpedia_14"
echo "=========================================="
python resume_steerability_test.py \
    --run_ids_pickle "$RUN_IDS_PICKLE" \
    --dataset "dbpedia_14" \
    --classifier_weight_suffixes "$CLASSIFIER_SUFFIXES" \
    --wandb_project "$WANDB_PROJECT" \
    $ENTITY_FLAG

echo "=========================================="
echo "Done running all steerability tests!"
echo "=========================================="
