python validateConvNext.py --img_size 128 --validate validate --split 100
#mv ./saved/predictLabels3Lval.pkl ./saved2/predictLabels3Ltrainval.pkl
mv ./saved/predictLabels3Lval.pkl ./saved_ConvNextV5/predictLabels3Ltrainval.pkl
python validateConvNext.py --img_size 128 --validate validate
#mv ./saved/predictLabels3Lval.pkl ./saved2/predictLabels3Lval.pkl
mv ./saved/predictLabels3Lval.pkl ./saved_ConvNextV5/predictLabels3Lval.pkl
python validateConvNext.py --img_size 128 --validate train
#mv ./saved/predictLabels3Ltrain.pkl ./saved2/predictLabels3Ltrain.pkl
mv ./saved/predictLabels3Ltrain.pkl ./saved_ConvNextV5/predictLabels3Ltrain.pkl

