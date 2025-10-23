# Conda environment for YOLO11 using Python 3.11 and tensorflow for flat classifier

conda create -n yolo11 python=3.11
conda activate yolo11
pip install -U scikit-image
pip install -U scikit-learn
# Only needed for the flat classifier model
# python -m pip install tensorflow[and-cuda]==2.17.0
pip install ultralytics
pip install pyqt5
pip install torchsummary
pip install pandas


