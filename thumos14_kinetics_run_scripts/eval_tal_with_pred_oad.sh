python tools/ontal_evaluator.py \
    --all_probs_file "models/reproduce_sequence_thumos14_decoder_layer4_kv7/validation_results/test_0_all_probs.pkl" \
    --ground_truth_cls "models/reproduce_sequence_thumos14_decoder_layer4_kv7/validation_results/test_0_ground_truth_cls.pkl" \
    --video_names_file "models/reproduce_sequence_thumos14_decoder_layer4_kv7/validation_results/test_0_video_names.pkl" \
    --ambiguos_info "/data/thumos14_feat/ambiguous_dict.pkl" \
    --oad_gt_file "/data/thumos14_feat/thumos_from_oad_multi_classes.pkl"
