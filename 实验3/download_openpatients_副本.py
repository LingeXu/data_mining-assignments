import os
from huggingface_hub import hf_hub_download

# 使用官方源（梯子已开）
os.environ.pop("HF_ENDPOINT", None)  # 移除清华镜像

local_dir = "../data/open-patients"
os.makedirs(local_dir, exist_ok=True)

# 只下载 48 MB 的 train.parquet
hf_hub_download(
    repo_id="ncbi/Open-Patients",
    repo_type="dataset",
    filename="train-00000-of-00001.parquet",
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print("下载完成，文件保存在：", os.path.abspath(local_dir))
