# SimOn: A Simple Framework for Online Temporal Action Localization

This is the official implementation of the paper [SimOn: A Simple Framework for Online Temporal Action Localization](). We provide training code for SimOn w/o context generation (P) as it is a part of our future research. The released code is for On-TAL and ODAS tasks on THUMOS14 dataset.

# SimOn: A Simple Framework for Online Temporal Action Localization

- This is the official implementation of the paper [SimOn: A Simple Framework for Online Temporal Action Localization](). We provide the code for training On-TAL and ODAS tasks, training log and model checkpoint on THUMOS14 dataset.

- For setting up the environment as well as dataset for training and testing, please checkout the `main` branch

### Train the model
- To train the model in ODAS task, use the following script:
```bash
./thumos14_annet_odas_run_scripts/train.sh
```

### Test the model
- To test the model in ODAS task, use the following script:
```bash
mkdir cache
./thumos14_annet_odas_run_scripts/test.sh
./thumos14_annet_odas_run_scripts/eval_tal_with_pred_odas.sh
```
- The model should give 54.0 mAP on average. Noted that the result may fluctuate a little.
