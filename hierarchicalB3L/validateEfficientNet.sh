python validateEfficientNet.py --img_size 128 --validate validate --split 100
mv ./saved_EfficientNetV6/predictLabels3Lval.pkl ./saved_EfficientNetV6/predictLabels3Ltrainval.pkl
python validateEfficientNet.py --img_size 128 --validate validate
python validateEfficientNet.py --img_size 128 --validate train
python testEfficientNet.py
