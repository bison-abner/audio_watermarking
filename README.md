# 富文本确权保护系统

## 项目简介

此项目为富文本确权保护系统，旨在通过音频和图片水印技术实现版权保护。系统提供了两个主要功能模块：音频确权模块和图片确权模块。用户可以在这些模块中添加或解码水印，确保内容的版权安全。

## 技术栈

- **Python**: 作为主要的编程语言。
- **Streamlit**: 用于创建和管理 Web 应用的界面。
- **PyTorch**: 深度学习框架，用于处理音频水印。
- **PIL (Python Imaging Library)**: 用于处理图片水印的图像库。
- **soundfile**: 用于读写音频文件。
- **numpy**: 提供高性能的数组操作。
- **datetime, uuid**: 用于生成带时间戳的唯一文件名。

## 功能说明

### 音频确权模块

- **添加水印**: 将水印信息嵌入音频文件中。
- **解码水印**: 从音频文件中提取水印信息。

### 图片确权模块

- **添加水印**: 在图片中嵌入水印信息。
- **解码水印**: 从图片中提取水印信息。

## 安装与运行

确保您的系统已安装 Python 和必要的依赖。

### 克隆仓库

```bash
git clone https://github.com/your-repository/audio_watermarking.git
cd audio_watermarking
```

## 如何上手使用

以下是使用该项目的简要步骤：

1. **环境准备：** 确保您的系统已安装 Python 和 PyTorch。
2. **下载项目：** 从项目仓库中下载项目文件，并解压到本地目录。
3. **安装依赖：** 安装项目所需的 Python 依赖库。
4. **运行项目：** 打开命令行，进入项目目录，并执行 `streamlit run app.py` 启动项目。
5. **使用功能：** 在项目界面中选择相应的功能模块，按照提示操作即可完成水印添加或解码的任务。

## 启动项目
**streamlit使用方法:**
每当您想要使用新环境时，首先需要转到项目文件夹（目录所在的位置.venv）并运行命令来激活它：
````
# Windows command prompt
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS and Linux
source .venv/bin/activate
````
激活后，您将在终端提示符开头的括号中看到环境名称。 “（.venv）”

**运行您的 Streamlit 应用程序:**

```streamlit run app.py```

**如果这不起作用，请使用长格式命令：**
```python -m streamlit run app.py```

要停止 Streamlit 服务器，请按Ctrl+C终端中的 。
(不要关浏览器，如果关了，先再运行项目，再关闭，不然关不上)

使用完此环境后，输入以下命令返回到普通 shell：

```deactivate```

## 常见问题

**altair包下载不下来**

### 步骤 1: 手动下载包

1. 访问 Anaconda.org：
   打开浏览器，访问 [Anaconda.org](https://anaconda.org)。

2. 搜索 altair 包：
   在搜索框中输入“altair”并按回车。在搜索结果中找到 altair 包，通常由多个版本供不同的操作系统使用。

3. 选择合适的版本：
   根据您的操作系统和 Python 版本选择合适的 altair 包。例如，如果您使用的是 Windows 且 Python 版本为 3.7，那么选择对应的 .tar.bz2 文件进行下载。

4. 下载包：
   点击所选包旁的下载图标，保存文件到您的计算机上。

### 步骤 2: 本地安装包

1. 在下载完 .tar.bz2 文件后，您可以通过 Conda 从本地文件安装该包：

2. 打开命令行：
   打开一个命令行窗口（Windows 中为 CMD 或 Anaconda Prompt）。

3. 导航到下载目录：
   使用 cd 命令导航到包下载的目录。例如：
   ```bash
   cd C:\Users\abner\Downloads
   ```

4. 使用 Conda 安装：
   运行以下命令以从本地文件安装 altair：
    ```bush 
    conda install altair-4.1.0-py37_0.tar.bz2
   ```
   
   请根据您下载的实际文件名替换 altair-4.1.0-py37_0.tar.bz2
    
### 步骤 3: 验证安装

1. 安装完成后，您可以通过以下命令验证 altair 是否正确安装
   ```bush 
    conda list altair
   ```
    这将列出已安装的 altair 包及其版本，确认安装成功。

    通过这些步骤，您应该能够在无法直接通过 Conda 命令或镜像源安装 altair 的情况下，
手动从 Anaconda.org 下载并安装所需的包。