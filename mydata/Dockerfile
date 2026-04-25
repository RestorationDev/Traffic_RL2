# Use CUDA-enabled base image
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Prevent tzdata and other packages from prompting
ENV DEBIAN_FRONTEND=noninteractive

# Set up Python and system dependencies
RUN apt-get update && apt-get install -y \
    tzdata \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    build-essential \
    cmake \
    && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 11.3
RUN pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install other Python libraries
RUN pip install numpy pandas matplotlib seaborn scipy tqdm networkx scikit-learn

# Clone and install CityFlow
RUN git clone https://github.com/cityflow-project/CityFlow.git /cityflow
WORKDIR /cityflow
RUN mkdir build && cd build && cmake .. && make -j$(nproc)

# Set PYTHONPATH safely (Docker will not interpolate this at build time)
ENV PYTHONPATH=/cityflow/build

# Set default work directory
WORKDIR /app

# Default shell
CMD ["bash"]


# FROM cityflowproject/cityflow:latest

# # Upgrade pip and install Python libraries
# RUN pip install --upgrade pip && \
#     pip install \
#         torch==1.10.0 \
#         torchvision \
#         numpy \
#         pandas \
#         matplotlib \
#         seaborn \
#         scipy \
#         tqdm \
#         networkx \
#         scikit-learn

# # Optional: set working directory
# WORKDIR /app

# # Optional: copy your project files
# # COPY . /app

# # Default to Python shell
# CMD ["bash"]