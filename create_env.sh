mamba create -n patcher python=3.9 tensorflow-gpu opencv matplotlib pandas scipy numpy pytorch torchvision \
     pytorch-cuda=11.6 -c pytorch -c nvidia
mamba activate patcher
pip install eckity

# YOLO:
pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt  # install dependencies

