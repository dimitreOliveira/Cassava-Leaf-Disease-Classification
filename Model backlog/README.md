## Model backlog
- **OOF** is the validation set using the training data from the competition.
- **Public LB** is the Public leaderboard score.
- **Private LB** is the Private leaderboard score.
- The competition metric is **Categorization Accuracy**.

---

## Models

| Model | OOF | Public LB | Private LB |
|-------|------------|-----------|------------|
| 0-Cassava Leaf-EfficientNetB3 TPU-v2 | 0.892 | 0.884 | ??? |
| 1-Cassava Leaf Disease-EfficientNetB3 TPU-v3 | 0.890 | 0.891 | ??? |
| 2-Cassava Leaf Disease-2020 data | 0.888 | 0.888 | ??? |
| 3-Cassava Leaf-2020 data oversampled | 0.888 | 0.883 | ??? |
| 4-Cassava Leaf-BN unfrozen | 0.886 | 0.881 | ??? |
| 5-Cassava Leaf-EfficientNetB5 | 0.893 | 0.890 | ??? |
| 6-Cassava Leaf-EfficientNetB5 456x456 | 0.892 | ??? | ??? |
| 7-Cassava Leaf-EfficientNetB5 light aug 456x456 | 0.889 | ??? | ??? |
| 8-Cassava Leaf-EfficientNetB5 light aug 512x512 | 0.893 | ??? | ??? |
| 9-Cassava Leaf-EfficientNetB5 15_epochs 512x512 | 0.893 | 0.896 | ??? |
| 10-Cassava Leaf-EfficientNetB5 2020+2019 512 | 0.895 | 0.893 | ??? |
| 11-Cassava Leaf-EfficientNetB5 2020 oversampled | 0.892 | 0.890 | ??? |
| 12-Cassava Leaf-EfficientNetB5 2020+2019 oversamp | 0.892 | 0.888 | ??? |
| 13-Cassava Leaf-EfficientNetB5 TFRec_15 512x512 | 0.897 | 0.889 | ??? |
| 14-Cassava Leaf-EfficientNetB5 smoothing_01 512 | 0.894 | 0.891 | ??? |
| 15-Cassava Leaf-EfficientNetB5 rotation 512x512 | 0.895 | 0.891 | ??? |
| 16-Cassava Leaf-EfficientNetB5 SGD 512 | 0.893 | 0.892 | ??? |
| 17-Cassava Leaf-EfficientNetB5 SGD 20_epochs 512 | 0.896 | 0.891 | ??? |
| 18-Cassava Leaf-EfficientNetB5 SGD 25_epochs 512 | 0.894 | 0.888 | ??? |
| 19-Cassava Leaf-EfficientNetB5 SGD LR finder 512 | 0.872 | 0.873 | ??? |
| 20-Cassava Leaf-EfficientNetB5 One cycle 512 | 0.893 | 0.890 | ??? |
| 21-Cassava Leaf-EfficientNetB5 One cycle cosine | 0.895 | 0.888 | ??? |
| 22-Cassava Leaf-EffNetB5 Cosine warmup LR 512 | 0.897 | 0.890 | ??? |
| 23-Cassava Leaf-EffNetB5 Cosine warmup LR 10c 512 | 0.900 | 0.893 | ??? |
| 24-Cassava Leaf-EffNetB5 no normalization 512 | 0.897 | 0.895 | ??? |
| 25-Cassava Leaf-EffNetB5 no normalization 10c 512 | 0.898 | 0.893 | ??? |
| 26-Cassava Leaf-EffNetB5 Adam Cosine warmup 512 | 0.897 | 0.889 | ??? |
| 27-Cassava Leaf-EffNetB5 RAdam Cosine warmup 512 | 0.895 | 0.894 | ??? |
| 28-Cassava Leaf-EffNetB5 Ranger Cosine warmup 512 | 0.895 | 0.889 | ??? |
| 29-Cassava Leaf-EffNetB5 2020 data 512 | 0.899 | 0.890 | ??? |
| 32-Cassava Leaf-EffNetB5 random_crop 456x456 | 0.896 | 0.894 | ??? |
| 33-Cassava Leaf-EffNetB5 456x456 | 0.898 | 0.891 | ??? |
| 34-Cassava Leaf-EffNetB5 Cosine 10c TPUv3 512 | 0.900 | 0.895 | ??? |
| 35-Cassava Leaf-EffNetB5 Cosine 10c TPUv2 512 | 0.901 | 0.897 | ??? |
| 36-Cassava Leaf-EffNetB5 dropout_025 512x512 | 0.899 | 0.893 | ??? |
| 37-Cassava Leaf-EffNetB5 dropout_025 512x512 | 0.898 | 0.893 | ??? |
| 38-Cassava Leaf-EffNetB5 rotation 512x512 | 0.899 | 0.891 | ??? |
| 39-Cassava Leaf-EffNetB5 better_crop 512x512 | 0.898 | 0.890 | ??? |
| 40-Cassava Leaf-EffNetB5 pixel-level 512x512 | 0.897 | 0.890 | ??? |
| 41-Cassava Leaf-EffNetB5 cut-out 512x512 | 0.900 | 0.891 | ??? |
| 42-Cassava Leaf-EffNetB5 2020_data cut-out 512x512 | 0.897 | 0.893 | ??? |
| 43-Cassava Leaf-EffNetB5 2020 pixel-level 512x512 | 0.897 | 0.890 | ??? |
| 44-Cassava Leaf-EffNetB5 2020 singcut-out 512x512 | 0.897 | 0.892 | ??? |
| 45-Cassava Leaf-EffNetB4 cut-out 512x512 | 0.894 | 0.886 | ??? |
| 46-Cassava Leaf-EffNetB5 adam 512x512 | 0.898 | 0.891 | ??? |
| 47-Cassava Leaf-EffNetB5 heavier aug 512x512 | 0.897 | 0.887 | ??? |
| 48-Cassava Leaf-EffNetB5 random_crop 456x456 | 0.892 | ??? | ??? |
| 49-Cassava Leaf-EffNetB5 random_crop_v2 456x456 | 0.896 | 0.891 | ??? |
| 50-Cassava Leaf-EffNetB5 center_crop 456 | 0.894 | 0.888 | ??? |
| 51-Cassava Leaf-EffNetB5 lbl_smooth_sched_inc 512 | 0.898 | 0.892 | ??? |
| 52-Cassava Leaf-EffNetB5 step_200 bs_8 512 | 0.895 | 0.891 | ??? |
| 53-Cassava Leaf-EffNetB5 step_200 bs_16 512 | 0.896 | 0.893 | ??? |
| 54-Cassava Leaf-EffNetB5 lbl_smooth_sched_dec 512 | 0.898 | 0.890 | ??? |
| 55-Cassava Leaf-EffNetB5 16-sample dropout 512 | 0.893 | 0.888 | ??? |
| 56-Cassava Leaf-EffNetB5 3_input_resize 512 | 0.886 | ??? | ??? |
| 57-Cassava Leaf-EffNetB5 3_input_random_crop 51 | 0.890 | ??? | ??? |
| 58-Cassava Leaf-EffNetB5 no_dropout 512 | 0.900 | 0.891 | ??? |
| 59-Cassava Leaf-EffNetB5 aux_task_healthy 512x512 | 0.894 | 0.889 | ??? |
| 60-Cassava Leaf-EffNetB5 aux_task_healthy smoo 512 | 0.897 | 0.890 | ??? |
| 61-Cassava Leaf-EffNetB5 aux_task_healthy smoo 512 | 0.899 | 0.895 | ??? |
| 62-Cassava Leaf-EffNetB5 aux_task_healthy_02 512 | 0.899 | 0.889 | ??? |
| 63-Cassava Leaf-EffNetB5 aux_task_cmd_02 512x512 | 0.901 | 0.892 | ??? |
| 64-Cassava Leaf-EffNetB5 aux_task_healt_cmd_02 512 | 0.898 | 0.891 | ??? |
| 65-Cassava Leaf-EffNetB5 aux_task_healt_cmd_01 512 | 0.898 | 0.895 | ??? |
| 66-Cassava Leaf-EffNetB5 5-Fold_41 512x512 | 0.900 | 0.893 | ??? |
| 67-Cassava Leaf-EffNetB5 5-Fold_61 512x512 | 0.900 | 0.894 | ??? |
| 68-Cassava Leaf-EffNetB5 5-Fold_63 512x512 | 0.899 | 0.896 | ??? |
| 69-Cassava Leaf-EffNetB5 5-Fold_51 512x512 | 0.899 | 0.895 | ??? |
| 70-Cassava Leaf-EffNetB5 complete_2019 512x512 | 0.898 | 0.890 | ??? |
| 71-Cassava Leaf-EffNetB5 2020_2019_oversampled 512 | 0.895 | 0.891 | ??? |
| 72-Cassava Leaf-EffNetB5 15_ep_cycle 512x512 | 0.897 | 0.894 | ??? |
| 73-Cassava Leaf-EffNetB5 5-Fold_63 TPU-v2 512x512 | 0.893 | 0.898 | ??? |
| 74-Cassava Leaf-EffNetB5 5-Fold_65 512x512 | 0.900 | 0.893 | ??? |
| 75-Cassava Leaf-EffNetB5 2_aux_tasks 2020_data 512 | 0.899 | 0.889 | ??? |
| 76-Cassava Leaf-BiT ResNet50x1 2_aux_tasks 512 | 0.893 | 0.888 | ??? |
| 77-Cassava Leaf-BiT ResNet50x1 flower_ exp aux 512 | 0.893 | 0.890 | ??? |
| 78-Cassava Leaf-BiT ResNet50x1 plant_exp aux 512 | 0.895 | 0.892 | ??? |
| 79-Cassava Leaf-BiT ResNet50x1 plant_exp adam 512 | 0.892 | 0.887 | ??? |
| 80-Cassava Leaf-BiT ResNet50x1 2_aux Adam 512 | 0.893 | 0.885 | ??? |
| 81-Cassava Leaf-EffNetB5 aux_task_2 bs_16 512 | 0.897 | 0.892 | ??? |
| 82-Cassava Leaf-EffNetB3 aux_task_2 SCL 512x512 | 0.881 | 0.881 | ??? |
| 83-Cassava Leaf-EffNetB3 SCL 2020_data 512x512 | 0.886 | 0.886 | ??? |
| 84-Cassava Leaf-EffNetB3 SCL 2020 2_aux 512x512 | 0.885 | ??? | ??? |
| 85-Cassava Leaf-EffNetB3 SCL 2020 2_aux SGD 512 | 0.829 | ??? | ??? |
| 86-Cassava Leaf-EffNetB4 SCL 2020 2_aux 512x512 | 0.883 | 0.891 | ??? |
| 87-Cassava Leaf-EffNetB5 SCL 2020 2_aux 512x512 | 0.882 | 0.886 | ??? |
| 88-Cassava Leaf-EffNetB5 cross_entropy 2_aux 512 | 0.892 | 0.888 | ??? |
| 89-Cassava Leaf-EffNetB4 cross_entropy 2_aux 512 | 0.892 | 0.886 | ??? |
| 90-Cassava Leaf-EffNetB4 s_cross_entropy aux 512 | 0.889 | 0.888 | ??? |
| 91-Cassava Leaf-EffNetB4 SCL aux 512x512 | 0.878 | 0.882 | ??? |
| 92-Cassava Leaf-EffNetB4 SCL+dense aux 512x512 | 0.888 | 0.892 | ??? |
| 93-Cassava Leaf-EffNetB4 SCL oversamp aux 512x512 | 0.874 | 0.873 | ??? |
| 94-Cassava Leaf-EffNetB4 SCL_no_aug oversamp 512 | 0.871 | 0.877 | ??? |
| 95-Cassava Leaf-EffNetB3 SCL oversample 512x512 | 0.882 | 0.880 | ??? |
| 96-Cassava Leaf-EffNetB3 SCL 3_proj oversample 512 | 0.878 | 0.871 | ??? |
| 97-Cassava Leaf-EffNetB3 SCL+CCE 512x512 | 0.882 | 0.882 | ??? |
| 98-Cassava Leaf-EffNetB3 SCL+CCE flat_proj 512x512 | 0.879 | ??? | ??? |
| 99-Cassava Leaf-EffNetB3 SCL+CCE 512x512 | 0.886 | ??? | ??? |
| 100-Cassava Leaf-EffNetB3 SCL+CCE BN_frozen 512 | 0.896 | 0.888 | ??? |
| 101-Cassava Leaf-EffNetB3 SCL+CCE BN SGD 512x512 | 0.886 | ??? | ??? |
| 102-Cassava Leaf-EffNetB3 SCL+CCE BN longer 512 | 0.894 | 0.893 | ??? |
| 103-Cassava Leaf-EffNetB3 SCL+CCE no_aug 512 | 0.896 | 0.886 | ??? |
| 104-Cassava Leaf-EffNetB3 SCL simple_class 512 | 0.892 | ??? | ??? |
| 105-Cassava Leaf-EffNetB3 SCL_3_enc_outputs 512 | 0.895 | 0.890 | ??? |
| 106-Cassava Leaf-EffNetB3 SCL_1_enc_outputs 512 | 0.895 | 0.888 | ??? |
| 107-Cassava Leaf-EffNetB3 SCL_enc_norm 512x512 | 0.895 | 0.884 | ??? |
| 108-Cassava Leaf-EffNetB3 SCL_enc_128 512 | 0.896 | 0.890 | ??? |
| 109-Cassava Leaf-EffNetB3 SCL_enc_aux 512x512 | 0.893 | 0.886 | ??? |
| 110-Cassava Leaf-EffNetB3 SCL 2019_data 512 | 0.893 | 0.888 | ??? |
| 111-Cassava Leaf-EffNetB3 SCL 2020_oversample 512 | 0.889 | 0.877 | ??? |
| 112-Cassava Leaf-EffNetB3 SCL_and_BCE 512 | 0.895 | 0.890 | ??? |
| 113-Cassava Leaf-EffNetB3 SCL less_dropout 512x512 | 0.896 | 0.888 | ??? |
| 114-Cassava Leaf-EffNetB4 SCL_enc_128 512x512 | 0.896 | 0.887 | ??? |
| 115-Cassava Leaf-EffNetB5 SCL_enc_128 512x512 | 0.895 | 0.890 | ??? |
| 116-Cassava Leaf-EffNetB3 SCL augment 512x512 | 0.894 | 0.888 | ??? |
| 117-Cassava Leaf-EffNetB3 SCL augment_pixel 512 | 0.880 | 0.872 | ??? |
| 118-Cassava Leaf-EffNetB3 SCL augment_medium 512 | 0.895 | 0.891 | ??? |
| 119-Cassava Leaf-EffNetB3 SCL augment_cutout 512 | 0.896 | 0.887 | ??? |
| 120-Cassava Leaf-EffNetB3 SCL augment_clip 512x512 | 0.894 | ??? | ??? |
| 121-Cassava Leaf-EffNetB3 SCL augment_heavy 512 | 0.895 | 0.887 | ??? |
