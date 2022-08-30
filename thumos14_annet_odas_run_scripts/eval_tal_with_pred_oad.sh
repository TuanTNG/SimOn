OUT_OADS="./cache/currrent.pkl"
rm $OUT_OADS
python tools/ontal_evaluator.py \
    --all_probs_file "models/odas_thumos_anet_decoder_layer4_kv7/validation_results/test_0_all_probs.pkl" \
    --ground_truth_cls "models/odas_thumos_anet_decoder_layer4_kv7/validation_results/test_0_ground_truth_cls.pkl" \
    --video_names_file "models/odas_thumos_anet_decoder_layer4_kv7/validation_results/test_0_video_names.pkl" \
    --ambiguos_info "/DATASET/thumos14_feat/ambiguous_dict.pkl" \
    --out_preds_odas $OUT_OADS \

# Eval ODAS
echo "Evaluating ODAS............................................"
ground_truth="/data/thumos14_feat/thumos14.json"
ontal_gt_from_cls_anno="/data/thumos14_feat/thumos_from_oad_multi_classes.pkl" # use this to get higher result
python tools/Evaluation/get_detection_performance.py \
    $ground_truth $OUT_OADS \
    --ontal_gt_from_cls_anno $ontal_gt_from_cls_anno
