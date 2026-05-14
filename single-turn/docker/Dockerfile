# ==== 基础镜像 ====
# 社区最新版本，含 CUDA 12.9 + PyTorch 2.8 + vLLM 0.11.0
FROM hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0

# ==== 基础设置 ====
WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
ENV VLLM_USE_V1=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Additional environment variables for the project
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
ENV MAX_JOBS=32

# ==== 安装必要工具 ====
RUN apt update && apt install -y git wget curl vim build-essential tini && \
    pip install --upgrade pip setuptools wheel && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ==== 安装 FlashAttention 2 ====
# （不重新编译 CUDA，直接兼容当前 PyTorch/CUDA 环境）
RUN pip install flash-attn --no-build-isolation

# ==== 安装 FlashInfer ====
# FlashInfer for efficient sampling (compatible with CUDA 12.8/12.9 and PyTorch 2.8)
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu128/torch2.8/ --no-build-isolation || \
    pip install flashinfer --no-build-isolation || \
    echo "FlashInfer installation optional, continuing..."

# ==== 复制并安装本地 Verl ====
# Copy all necessary files for verl installation
COPY setup.py pyproject.toml README.md LICENSE Notice.txt /workspace/verl_project/
COPY verl /workspace/verl_project/verl
RUN cd /workspace/verl_project && \
    pip install -e .[vllm,fsdp,megatron] && \
    cd /workspace && \
    rm -rf /root/.cache/pip

# Create symbolic link for easier access
RUN ln -s /workspace/verl_project/verl /workspace/verl

# ==== 安装 Megatron-LM ====
# Note: Using main branch as recommended versions may not have release tags
RUN cd /workspace && \
    if [ -d "Megatron-LM" ]; then rm -rf Megatron-LM; fi && \
    git clone https://github.com/NVIDIA/Megatron-LM.git && \
    cd Megatron-LM && \
    git checkout core_r0.10.0 && \
    pip install -e . || echo "Megatron-LM installation optional, continuing..."

# ==== 安装项目额外依赖 ====
RUN pip install --no-cache-dir \
    codetiming \
    latex2sympy2_extended \
    math_verify \
    qwen-vl-utils \
    tensorboard

# ==== 验证依赖 ====
RUN python -c "import torch, vllm; print('Torch:', torch.__version__, 'vLLM:', vllm.__version__)" && \
    python -c "import torch.distributed.fsdp; print('FSDP backend ready')" && \
    python -c "import flash_attn; print('FlashAttention:', flash_attn.__version__)"

# ==== 创建项目目录 ====
RUN mkdir -p /workspace/logs /workspace/outputs /workspace/data /workspace/.cache/huggingface

# ==== 复制项目代码 ====
COPY . /workspace/project
WORKDIR /workspace/project

# ==== 设置项目环境变量 ====
ENV VERL_PATH=/workspace/verl_project
ENV WORK_DIR=/workspace/project
ENV PYTHONPATH=/workspace/verl_project:/workspace/project:${PYTHONPATH}
ENV HF_HOME=/workspace/.cache/huggingface

# ==== 暴露端口 ====
# Ray dashboard and TensorBoard
EXPOSE 8265 6006

# ==== 使用 tini 作为入口点 ====
ENTRYPOINT ["/usr/bin/tini", "--"]

# ==== 默认命令 ====
CMD ["/bin/bash"]

