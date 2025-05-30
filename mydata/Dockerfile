FROM cityflowproject/cityflow:latest

# Upgrade pip and install Python libraries
RUN pip install --upgrade pip && \
    pip install \
        torch==1.10.0 \
        torchvision \
        numpy \
        pandas \
        matplotlib \
        seaborn \
        scipy \
        tqdm \
        networkx \
        scikit-learn

# Optional: set working directory
WORKDIR /app

# Optional: copy your project files
# COPY . /app

# Default to Python shell
CMD ["python"]