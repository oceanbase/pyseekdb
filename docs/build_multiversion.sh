#!/bin/bash
# 构建多版本文档的脚本

set -e

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}开始构建多版本文档...${NC}"

# 确保在项目根目录
cd "$(dirname "$0")/.."

# 激活虚拟环境（如果存在）
if [ -d ".venv" ]; then
    echo -e "${GREEN}激活虚拟环境...${NC}"
    source .venv/bin/activate
fi

# 清理旧的构建文件
echo -e "${GREEN}清理旧的构建文件...${NC}"
rm -rf docs/_build/html

# 使用 sphinx-multiversion 构建文档
echo -e "${GREEN}使用 sphinx-multiversion 构建文档...${NC}"
sphinx-multiversion docs docs/_build/html

# 创建重定向到最新版本的 index.html
echo -e "${GREEN}创建重定向页面...${NC}"
cat > docs/_build/html/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
  <title>Redirecting to latest version</title>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=./develop/index.html">
  <link rel="canonical" href="./develop/index.html">
</head>
<body>
  <p>Redirecting to <a href="./develop/index.html">latest version</a>...</p>
</body>
</html>
EOF

echo -e "${BLUE}多版本文档构建完成！${NC}"
echo -e "${GREEN}文档位置: docs/_build/html${NC}"
echo -e "${GREEN}可以使用以下命令启动本地服务器查看:${NC}"
echo -e "  cd docs/_build/html && python -m http.server 8000"
