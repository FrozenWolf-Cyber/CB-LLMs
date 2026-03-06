#!/bin/bash
# Train 3 RoBERTa classifiers with different seeds for steerability evaluation
# Hardcoded for all 4 datasets

set -e

# ===================== SetFit/sst2 =====================
echo "========================================"
echo "Training classifiers for SetFit/sst2"
echo "========================================"
python train_classifier.py --dataset "SetFit/sst2" --seed 42 --weight_suffix "_seed42"
python train_classifier.py --dataset "SetFit/sst2" --seed 123 --weight_suffix "_seed123"
python train_classifier.py --dataset "SetFit/sst2" --seed 456 --weight_suffix "_seed456"

# ===================== ag_news =====================
echo "========================================"
echo "Training classifiers for ag_news"
echo "========================================"
python train_classifier.py --dataset "ag_news" --seed 42 --weight_suffix "_seed42"
python train_classifier.py --dataset "ag_news" --seed 123 --weight_suffix "_seed123"
python train_classifier.py --dataset "ag_news" --seed 456 --weight_suffix "_seed456"

# ===================== yelp_polarity =====================
echo "========================================"
echo "Training classifiers for yelp_polarity"
echo "========================================"
python train_classifier.py --dataset "yelp_polarity" --seed 42 --weight_suffix "_seed42"
python train_classifier.py --dataset "yelp_polarity" --seed 123 --weight_suffix "_seed123"
python train_classifier.py --dataset "yelp_polarity" --seed 456 --weight_suffix "_seed456"

# ===================== dbpedia_14 =====================
echo "========================================"
echo "Training classifiers for dbpedia_14"
echo "========================================"
python train_classifier.py --dataset "dbpedia_14" --seed 42 --weight_suffix "_seed42"
python train_classifier.py --dataset "dbpedia_14" --seed 123 --weight_suffix "_seed123"
python train_classifier.py --dataset "dbpedia_14" --seed 456 --weight_suffix "_seed456"

echo "========================================"
echo "Done training all classifiers!"
echo "========================================"
