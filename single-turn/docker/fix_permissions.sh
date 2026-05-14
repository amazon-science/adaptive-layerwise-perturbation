#!/bin/bash
# 权限修复脚本 - 在启动Docker前运行
# 确保Docker可以访问所有挂载目录

echo "==========================================="
echo "修复 Docker 挂载目录权限..."
echo "==========================================="

# 确保home目录有执行权限（允许遍历）
echo "1. 设置 /home/zhang430 权限为 711 (rwx--x--x)..."
chmod 711 /home/zhang430 2>/dev/null || echo "  注意: 无法修改 /home/zhang430 权限（可能需要管理员权限）"

# 确保code目录可访问
echo "2. 设置 /home/zhang430/code 权限为 755 (rwxr-xr-x)..."
chmod 755 /home/zhang430/code 2>/dev/null || echo "  注意: 无法修改 /home/zhang430/code 权限"

# 确保项目目录可访问
echo "3. 设置 /home/zhang430/code/mismatch_rl 权限为 755..."
chmod 755 /home/zhang430/code/mismatch_rl 2>/dev/null || echo "  注意: 无法修改项目目录权限"

# 确保data目录可访问
echo "4. 设置 /home/zhang430/data 权限为 755..."
chmod 755 /home/zhang430/data 2>/dev/null || echo "  注意: 无法修改 data 目录权限"

# 创建并设置checkpoints目录（完全读写）
echo "5. 创建并设置 checkpoints 目录权限..."
mkdir -p /home/zhang430/checkpoints/mismatch_rl_research 2>/dev/null
chmod -R 777 /home/zhang430/checkpoints 2>/dev/null || echo "  注意: 无法设置 checkpoints 权限"

# 创建并设置outputs和logs目录（完全读写）
echo "6. 创建并设置 outputs/logs 目录权限..."
mkdir -p /home/zhang430/code/mismatch_rl/outputs /home/zhang430/code/mismatch_rl/logs 2>/dev/null
# 只设置目录本身的权限，忽略内部文件的权限错误
chmod 777 /home/zhang430/code/mismatch_rl/outputs 2>/dev/null || true
chmod 777 /home/zhang430/code/mismatch_rl/logs 2>/dev/null || true

# HuggingFace cache
echo "7. 创建并设置 HuggingFace 缓存目录权限..."
mkdir -p /home/zhang430/code/mismatch_rl/.cache/huggingface 2>/dev/null
chmod -R 777 /home/zhang430/code/mismatch_rl/.cache/huggingface 2>/dev/null || echo "  注意: 无法设置 HuggingFace 缓存目录权限"

echo ""
echo "==========================================="
echo "权限检查结果:"
echo "==========================================="
echo "/home/zhang430:                 $(ls -ld /home/zhang430 | awk '{print $1, $3, $4}')"
echo "/home/zhang430/code:            $(ls -ld /home/zhang430/code | awk '{print $1, $3, $4}')"
echo "/home/zhang430/code/mismatch_rl:$(ls -ld /home/zhang430/code/mismatch_rl | awk '{print $1, $3, $4}')"
echo "/home/zhang430/data:            $(ls -ld /home/zhang430/data | awk '{print $1, $3, $4}')"
echo "/home/zhang430/checkpoints:     $(ls -ld /home/zhang430/checkpoints | awk '{print $1, $3, $4}')"
echo ""
echo "✓ 权限设置完成！"
echo "==========================================="

