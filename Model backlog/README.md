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
| 43-Cassava Leaf-EffNetB5 2020 pixel-level 512x512 | 0.897 | ??? | ??? |
| 44-Cassava Leaf-EffNetB5 2020 singcut-out 512x512 | 0.897 | ??? | ??? |
