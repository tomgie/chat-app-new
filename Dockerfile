FROM runpod/base:0.6.2-cuda12.2.0

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb

RUN apt-get update && apt-get upgrade -y

# Install various needed dependencies
RUN apt-get install -y \
    python3

# Install CUDA dependencies
#RUN apt-get install -y \
#    cuda-toolkit \
#    cudnn9-cuda-12

# Update Python pip
RUN python3 -m pip install --upgrade pip

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "fastapi[standard]"

CMD ["fastapi", "run", "main.py"]