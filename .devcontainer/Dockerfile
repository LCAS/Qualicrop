# Base image with PyTorch, CUDA, and cuDNN
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev python3-tk \
    git wget curl unzip build-essential \
    libopencv-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --verbose\
    numpy scipy pandas matplotlib seaborn \
    scikit-learn spectral opencv-python plotly \
    scikit-image ultralytics \
    h5py
# Install TensorFlow with CUDA 11.8 support
RUN python3 -m pip install tensorflow==2.12.0

# Install PyTorch with CUDA 11.8 support
RUN python3 -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Create a directory for additional packages and clone repositories
RUN mkdir -p /workspace/packages && \
    cd /workspace/packages 
    #cd /workspace/packages && \
    #git clone https://github.com/MahmoudYidi/Anom_GUI.git && \
    #git clone https://github.com/MahmoudYidi/YOLOv8_preprocess.git

    




#GUI
COPY ../.devcontainer/GUI.sh /usr/local/bin/gui_support.sh
RUN chmod +x /usr/local/bin/gui_support.sh
# Copy verify script into the container
COPY ../src/verify.py /workspace/src/verify.py


WORKDIR /workspace

# Default command
#CMD ["/usr/local/bin/gui_support.sh", "/bin/bash"]
CMD ["/usr/local/bin/gui_support.sh", "/bin/bash"]