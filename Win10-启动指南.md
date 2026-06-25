# Windows 10 启动指南

> 本文档针对 Deep RAG 项目在 Win10 上的启动说明。
> 项目自带的 `start.sh` / `stop.sh` / `restart.sh` 均为 Bash 脚本,**无法直接在 Win10 上运行**,需按以下步骤手动启动。

---

## 📋 项目概览

Deep RAG 是一个"深度检索增强生成"系统,包含两部分:

| 模块 | 技术栈 | 端口 |
|------|--------|------|
| 后端 | Python + FastAPI + Uvicorn | 8000 |
| 前端 | React + TypeScript + Vite | 5173 |

项目根目录:`e:\cursor_workspace\deep-rag`

---

## 🔧 前置要求

在命令行分别运行以下命令确认已安装:

```bash
python --version    # 需 3.8+
node --version      # 需 16+
npm --version
```

如未安装:
- Python: https://www.python.org/downloads/
- Node.js: https://nodejs.org/

---

## ✅ 当前环境状态(已检测)

| 检查项 | 状态 |
|--------|------|
| `.env` 配置文件 | ✅ 已存在 |
| Python 虚拟环境 `venv` | ✅ 已创建 |
| 前端依赖 `node_modules` | ❌ **未安装,需先执行 npm install** |

---

## 🚀 启动步骤

> 建议**打开两个终端窗口**(PowerShell 或 CMD),分别运行后端和前端。

### 第 1 步:配置 `.env`

打开项目根目录的 `.env` 文件,确认以下两项:

1. `API_PROVIDER` 设置为你想用的服务商(如 `google` / `openai` / `anthropic` / `custom`)
2. 对应服务商的 `*_API_KEY` 已填入**真实密钥**(不是 `your_xxx_key` 占位符)

示例(使用 Google Gemini):

```bash
API_PROVIDER=google
GOOGLE_API_KEY=你的真实key
GOOGLE_MODEL=gemini-2.5-flash-lite
```

完整配置项说明见 [.env.example](.env.example)。

---

### 第 2 步:安装前端依赖(首次必做)

在项目根目录执行:

```bash
cd e:\cursor_workspace\deep-rag\frontend
npm install
cd ..
```

> 后续如果 `package.json` 有变更,再重新执行一次即可。

---

### 第 3 步:启动后端(终端 1)

```bash
cd e:\cursor_workspace\deep-rag
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

看到类似下面的输出即成功:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

- 激活虚拟环境后,命令行提示符前会出现 `(venv)` 字样。
- `pip install` 用于同步依赖,已装过会很快跳过。

---

### 第 4 步:启动前端(终端 2,新开窗口)

```bash
cd e:\cursor_workspace\deep-rag\frontend
npm run dev
```

看到类似输出即成功:

```
  VITE v7.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
```

---

### 第 5 步:访问应用

浏览器打开:**http://localhost:5173**

- 前端 UI:http://localhost:5173
- 后端 API:http://localhost:8000

---

## 🛑 停止服务

在对应终端窗口按 `Ctrl + C` 即可停止前端或后端。

---

## 🔁 重启服务

- **重启后端**:在后端终端 `Ctrl + C` 后,重新执行第 3 步的最后一条 `python -m uvicorn ...` 命令即可(虚拟环境已激活,无需重装依赖)。
- **重启前端**:Vite 默认支持热更新,改前端代码一般无需手动重启;如确需重启,`Ctrl + C` 后重新 `npm run dev`。

---

## ❓ 常见问题

### Q1:`venv\Scripts\activate` 提示无法执行脚本(执行策略错误)

PowerShell 默认禁止运行脚本。以**管理员身份**打开 PowerShell 执行一次:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

然后重试激活命令。或改用 CMD 运行。

### Q2:`npm install` 很慢或失败

切换为国内镜像:

```bash
npm config set registry https://registry.npmmirror.com
```

再重新 `npm install`。

### Q3:`pip install` 很慢或失败

切换为国内源:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q4:端口被占用

- 后端 8000 被占:修改启动命令中的 `--port 8000` 为其他端口(如 8001),同时需检查前端是否硬编码了后端地址。
- 前端 5173 被占:Vite 会自动顺延到 5174 等端口,按终端输出的实际地址访问即可。

### Q5:前端页面打不开 / 报错连接后端失败

- 确认**后端终端**已成功启动并显示 `Uvicorn running`。
- 检查 `.env` 中的 `API_PROVIDER` 和 API Key 配置是否正确。

---

## 📂 关键文件速查

| 文件 | 作用 |
|------|------|
| [.env](.env) | 环境变量配置(API Key、模型、提供商等) |
| [.env.example](.env.example) | 配置模板(完整字段说明) |
| [requirements.txt](requirements.txt) | Python 依赖清单 |
| [frontend/package.json](frontend/package.json) | 前端依赖清单 |
| [backend/main.py](backend/main.py) | 后端入口 |
| [backend/config.py](backend/config.py) | 配置加载逻辑 |
| start.sh / stop.sh / restart.sh | Linux/macOS 用脚本(Win10 不可用) |
