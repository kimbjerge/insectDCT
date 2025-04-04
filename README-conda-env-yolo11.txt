# Conda environment for YOLO11 using Python 3.11 and tensorflow for classifier

conda create -n yolo11 python=3.11
conda activate yolo11
pip install -U scikit-image
pip install -U scikit-learn
python -m pip install tensorflow[and-cuda]==2.17.0
pip install ultralytics
pip install pyqt5
pip install torchsummary

