This is a step-by-step guide helping you run MindSpore models in SGLang.

### 1. Install CANN

Please install the 8.3.RC1 community edition: [https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1]
This step includes Ascend toolkit, kernel and nnal, as well as pip installing te and hccl.

Assume the installation path is `/usr/local/Ascend`.

### 2. Install SGLang with Ascend support

```
git clone https://github.com/sgl-project/sglang.git
cd sglang
cp python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_npu]"
```

### 3. Install MindSpore and other required packages

```
pip install mindspore==2.7.1 torch==2.8.0 torch_npu==2.8.0 torchvision==0.23.0 triton-ascend pybind11
```

### 4. Build and install sgl-kernel-npu

This step requires GCC version >= 9.

```
git clone https://github.com/sgl-project/sgl-kernel-npu.git
cd sgl-kernel-npu
bash build.sh -a kernels
pip install output/*.whl
```

### 5. Install MindSpore models repo
```
git clone https://github.com/chz34/sgl-mindspore.git
cd sgl-mindspore
pip install -e .
```

### 6. Rename the nnal directory

This prevents conflict of ATB symbol links in MindSpore and torch-npu. This step will no longer be needed in future versions of MindSpore.

```
mv usr/local/Ascend/nnal usr/local/Ascend/nnal_
```
