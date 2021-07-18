# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.05-py3

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache -r requirements.txt coremltools onnx gsutil notebook
# RUN pip install --no-cache -U torch torchvision

# # Create working directory
# RUN mkdir -p /usr/src/app
# WORKDIR /usr/src/app

# # Copy contents
# COPY . /usr/src/app

# # Set environment variables
# ENV HOME=/usr/src/app
