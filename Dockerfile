FROM runpod/base:0.6.2-cuda12.2.0

# Install various needed dependencies
RUN apt-get install -y \
    python3-dev

# Install CUDA dependencies
RUN apt-get install -y \
    cuda-toolkit \
    cudnn9-cuda-12

RUN python -m pip install --upgrade pip

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "fastapi[standard]"

CMD ["fastapi run main.py"]