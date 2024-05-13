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

```
audio_watermarking
├─ -gitignore.url
├─ .git
│  ├─ COMMIT_EDITMSG
│  ├─ config
│  ├─ description
│  ├─ FETCH_HEAD
│  ├─ HEAD
│  ├─ hooks
│  │  ├─ applypatch-msg.sample
│  │  ├─ commit-msg.sample
│  │  ├─ fsmonitor-watchman.sample
│  │  ├─ post-update.sample
│  │  ├─ pre-applypatch.sample
│  │  ├─ pre-commit.sample
│  │  ├─ pre-merge-commit.sample
│  │  ├─ pre-push.sample
│  │  ├─ pre-rebase.sample
│  │  ├─ pre-receive.sample
│  │  ├─ prepare-commit-msg.sample
│  │  ├─ push-to-checkout.sample
│  │  └─ update.sample
│  ├─ index
│  ├─ info
│  │  └─ exclude
│  ├─ logs
│  │  ├─ HEAD
│  │  └─ refs
│  │     ├─ heads
│  │     │  └─ master
│  │     └─ remotes
│  │        └─ origin
│  │           └─ master
│  ├─ objects
│  │  ├─ 00
│  │  │  └─ 1adaf2afd7caf65cd7fea6ee991783685d54ca
│  │  ├─ 0b
│  │  │  └─ 968aaedad736d5561561f2b6f06136ac12dc67
│  │  ├─ 10
│  │  │  └─ 5ce2da2d6447d11dfe32bfb846c3d5b199fc99
│  │  ├─ 13
│  │  │  └─ 566b81b018ad684f3a35fee301741b2734c8f4
│  │  ├─ 18
│  │  │  ├─ 247337babdb18832ea769edd760f7652c070d7
│  │  │  └─ dd43b4c193c4205a66561c3c6d76860888bc1e
│  │  ├─ 1e
│  │  │  └─ e8d515b430517c5f96ac1db901a5b3c887af59
│  │  ├─ 22
│  │  │  └─ 6a9ce235c01d7faddc1c3d9c42a0bbdd3adcf6
│  │  ├─ 27
│  │  │  └─ 487cac5f5eb66b7a62e3318839330ee474d4ed
│  │  ├─ 2c
│  │  │  └─ 80e1269497d12e018fd6afa29982e56b0fb70d
│  │  ├─ 31
│  │  │  └─ cb4a39d1b2b3f083fe2064ffa5bef9845a2780
│  │  ├─ 39
│  │  │  └─ 3863463f8aac208bdbbc5aa1ac420bb6c710db
│  │  ├─ 3d
│  │  │  └─ b7543883e9370fcc8ba05034dd8ae9b81b69fe
│  │  ├─ 41
│  │  │  └─ 3b74cced356d37514d45df3217a8478fd03a0f
│  │  ├─ 49
│  │  │  └─ 2a13336ef4233ee081e803e474f94edd8656f2
│  │  ├─ 50
│  │  │  └─ 0059d602d30d52983ab4496b6dbbd5dba217b0
│  │  ├─ 53
│  │  │  └─ 891ec5fabe74db7b97b8718e9ac066933c75cf
│  │  ├─ 55
│  │  │  └─ 96b44786f04e4810aefe9f8d712f08ed310f71
│  │  ├─ 68
│  │  │  └─ bc17f9ff2104a9d7b6777058bb4c343ca72609
│  │  ├─ 6c
│  │  │  └─ 66e234f88100b4e6d4682480d9ba0787d61d38
│  │  ├─ 6d
│  │  │  └─ be3f794a02ca1574853450ff98617c5d781555
│  │  ├─ 6e
│  │  │  └─ d1947b70edbda356958b533168578ab7740ab1
│  │  ├─ 6f
│  │  │  ├─ 1ee150582bf44de4249b73540c97e1e7bfbbcf
│  │  │  └─ f7e72030ecb0ed969b3baec0986b91336670f8
│  │  ├─ 72
│  │  │  ├─ 6c5938c3a945bf1ec1a1922df9581bbb8c5203
│  │  │  └─ 836461a4004adeb919b0c910742f134c967b05
│  │  ├─ 73
│  │  │  └─ 206e6b18325e12c048822e7fa95c91b3756989
│  │  ├─ 78
│  │  │  └─ 4e29a14d25be542f778fecd6c85b3105479439
│  │  ├─ 7c
│  │  │  └─ 16d88f92a6fef775104929cb7c5587931927a3
│  │  ├─ 87
│  │  │  └─ cd77e19bdf04780a98806fff826655967107ae
│  │  ├─ 89
│  │  │  └─ 2ae02dd43cd5cde45168ed6c905e84b48afbe8
│  │  ├─ 8a
│  │  │  └─ ac2fe965e7c14c3c5b64cec14feb74dd96d889
│  │  ├─ 8b
│  │  │  └─ 2914e631c3d19237308e3b2cb5e3afa6ace5ab
│  │  ├─ 8d
│  │  │  └─ 1f22cc19e13dac89a01fc53ba1527d4f459ec8
│  │  ├─ 94
│  │  │  └─ a25f7f4cb416c083d265558da75d457237d671
│  │  ├─ 97
│  │  │  ├─ 094fe2847704a299ad9e738ec20a64d0dc0dac
│  │  │  └─ 824ec9b0149e48c91dce4fee530335b327c686
│  │  ├─ 9b
│  │  │  ├─ 521f3777b2aa38a8863674c663fb4f88e6b278
│  │  │  └─ 52b5d040f29ae6af1dbe353a60a50c4412150f
│  │  ├─ a4
│  │  │  └─ dac014f229fca0aa2e735ef8927f4ad0b88b43
│  │  ├─ a5
│  │  │  ├─ d8e02ff5f832309bfd261f7cd3d0588e913030
│  │  │  └─ e3e8a216dfd54e9e8eb76452596aa42d109851
│  │  ├─ b4
│  │  │  └─ ce554aa0e8a92b1b5950c6420a0b54d403676a
│  │  ├─ bc
│  │  │  ├─ 1df13c15ca5dcc247e7210b1c484ab2a6cd4a7
│  │  │  ├─ 24f260b0d26a239c9334458bacf625bc4d0980
│  │  │  ├─ ee787b36dca14c2df8762ce42d1b28bc92a1d4
│  │  │  └─ f44b630b1406a367c86a31db67bed93fbb3ced
│  │  ├─ c0
│  │  │  └─ 0c3b92db5dd5098de5f523fb30836dfd26b356
│  │  ├─ c2
│  │  │  └─ 8351cbc602370e194a31cca911bca645111f9a
│  │  ├─ c6
│  │  │  └─ 1cc15a7dba3c2ae7883a1e1186523b1d1fd7d6
│  │  ├─ c8
│  │  │  └─ e029fbf5539255dcfbc12979da89f8868ca1ab
│  │  ├─ d4
│  │  │  └─ 7a02988d1501d5c49053a9e75aa47d92c0e5d2
│  │  ├─ d7
│  │  │  └─ ea3c7c734938f9d1a0f91523dddb7bcf71af65
│  │  ├─ da
│  │  │  └─ 87f061cfc16b8ee7bdd4879d576777d3925c4c
│  │  ├─ dc
│  │  │  └─ eb6025fdefc01e355f679d00da377029738c84
│  │  ├─ de
│  │  │  └─ 21caffa5b965aab8b4d09e75eb69b2e0db0dc1
│  │  ├─ df
│  │  │  └─ 765d626470de5f9afc5273d750953709be58bf
│  │  ├─ e6
│  │  │  └─ 9de29bb2d1d6434b8b29ae775ad8c2e48c5391
│  │  ├─ ea
│  │  │  └─ ecef652cd9540bdd09d519e518e60b232c7804
│  │  ├─ f4
│  │  │  └─ 0d4f337bb04a0e03542ee6b5cb86f6be1c779b
│  │  ├─ f8
│  │  │  └─ 5393d9a2fd65457dad53a9a285497927ba06a2
│  │  ├─ info
│  │  └─ pack
│  ├─ ORIG_HEAD
│  └─ refs
│     ├─ heads
│     │  └─ master
│     ├─ remotes
│     │  └─ origin
│     │     └─ master
│     └─ tags
├─ .gitignore
├─ .idea
│  ├─ .gitignore
│  ├─ .name
│  ├─ inspectionProfiles
│  │  └─ profiles_settings.xml
│  ├─ misc.xml
│  ├─ modules.xml
│  ├─ RTR_protection_system.iml
│  ├─ vcs.xml
│  └─ workspace.xml
├─ app.py
├─ blind watermark
│  ├─ blind watermark.py
│  ├─ bwm_core.py
│  └─ pool.py
├─ LICENSE
├─ LICENSE.url
├─ models
│  ├─ hinet.py
│  ├─ invblock.py
│  ├─ module_util.py
│  ├─ my_model_v7_recover.py
│  ├─ rrdb_denselayer.py
│  └─ __init__.py
├─ README.md
├─ README.md.url
├─ requirements.txt
├─ step59000_snr39.99_pesq4.35_BERP_none0.30_mean1.81_std1.81.pkl
└─ utils
   ├─ bin_util.py
   ├─ file_reader.py
   ├─ metric_util.py
   ├─ model_util.py
   ├─ pesq_util.py
   ├─ pickle_util.py
   ├─ silent_util.py
   ├─ wm_add_v2.py
   ├─ wm_decode_v2.py
   └─ __init__.py

```