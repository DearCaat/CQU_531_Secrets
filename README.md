# 重大软院531实验室秘籍

我发现在研究生学习中遇到的很多问题都是相似的，不管是科研任务、代码实现、服务器管理。但一届一届的师兄师弟都在找着同样的资料，浪费着同样的时间。我希望这种经验贴能够帮助新老同学，新同学能够更快入门，老同学没必要去纠结自己一年前解决过的问题。个人能力有限，希望同学们一起分享、订正、补充 (留下Issues或者直接联系我) 🤗。

**Notes:** 很多内容没有具体详细的代码和指令，只会提供一个思路或者关键词，希望同学们能够善用**ChatGPT**、**Bing AI**、**Google** 😀。

**Tips:**

- **代码问题**：优先查看相应的官方文档、Github官方仓库的Issues（很多软件或者包都是开源的，都有官方仓库，***特别是当你复现人家论文的时候，找到作者的仓库，然后直接看Issues或者提出Issues***）、Bing AI
- **论文问题**：最好看原文，再结合官方代码。知乎或者CSDN解析都差很多意思，而且最新的文章一般没有（该建议来自于其它大佬）
- **Latex问题**：Bing AI、Google、Overleaf（Overleaf上其实有很多很多Latex相关的语法或者编译问题解答，文档较全，一般能解决问题）
- **Adobe破解**：https://baiyunju.cc/8602 （一般来说老的版本可以用[amt-emulator](https://amtemu-official.com/amtemu-v0-9-2-patcher/)，去破解`amtlib.ddl`，只要包含`amtlib.ddl`的版本应该都能破解，不仅仅是2017及以前。**但是vposy大神的版本感觉更好用更安全点，就是可能百度网盘下载速度有点问题，[这里](https://baiyunju.cc/10362)也有个妥协的办法**）
- **实验问题**：推荐大家使用**Docker、Wandb、Git**管理自己的实验环境、数据、代码、模型。这可能是完成高质量实验的一个基础，有利于你实验结果的稳定性、可复现性、可扩展性 （***据我了解，在大组、公司这几乎是必须的流程***）

**Resources:**

- **wandb**： http://10.236.11.202:8080. 用户名和密码私下找我
- **docker registry**: http://10.236.11.202:5000. 用户名和密码私下找我
- **New Bing (破解版)**: http://10.236.11.67:3001/

**Status:** 整理中~~~~~~~



## Table of Contents

- [工具](#工具推荐)
- [论文]()
  - [如何找论文](#如何找论文)
  - [如何写Pytoch格式的伪代码](#如何用latex-写-pytorch-pseudocode)
  - [如何插入多个参考文献小节](#如何在latex中引入多个bib)
  - [如何找到论文复现代码](#如何找到论文复现代码较为重要)
- [实验]()
  - [Docker相关]()
    - [如何在Rootless的情况下使用Docker](#如何在rootless的情况下使用docker)
    - [如何在Docker中使用CUDA](#docker-支持cuda)
    - [如何搭建私有仓库](#搭建私有docker-registry)
    - [如何用Docker跑实验（持续更新中）](#用docker跑实验)
  - [Linux相关（Ubuntu）]()
    - [如何重装Ubuntu](#如何重装ubuntu)
    - [如何在线更新Ubuntu](#在线升级ubuntu服务器)
    - [如何装Nvidia3件套，nvidia-smi相关](#安装nvidia驱动)
    - [如何给Ubuntu翻墙](#ubuntu配置clash)
    - [如何挂载移动硬盘](#ubuntu下挂载移动硬盘并修改权限)
    - [跨网段传输文件，ssh连接](#跨网段ssh连接)
  - [如何给服务器联网（dogcom）](#如何给服务器联网dogcom)
  - [如何在服务器上写代码（VSCode）](#如何在服务器上写代码vscode)
  - [如何用更少显存更快更好跑实验 （AMP）](#如何用更少显存更快更好跑实验-amp)
  - [如何管理实验数据（使用Wandb+Git）](#如何管理实验数据-使用wandbgit)
  - [如何管理wandb服务器](#重新创建app)
  - [如何远程操作实验室电脑](#如何远程操作实验室电脑)
  - [如何抢卡](#如何抢卡慎用慎用慎用)
  - [如何在多个服务器之间同步代码](#如何在多个服务器之间同步代码)
  - [如何将Tensorflow模型转Pytorch](#如何将keras模型-tensorflow-转torch)
  - [如何仅让部分IP通过指定VPN](#如何仅让部分ip通过指定vpnwindows--默认vpn-修改路由)
  - [如何在服务器之间或服务器与本地传输数据集](#如何传输数据集)
  - [显卡驱动掉了如何重装](#如何重装驱动)

## 工具推荐

工欲善其事必先利其器

### ReadPaper（论文管理）

> 我尝试使用过Zotero这种自定义更高的工具，但**ReadPaper**相关功能更易用且更完善。个人推荐。

### Edge浏览器（多端同步）

> 可以在设置里面选择下载预览版（`dev`版），一般来说可以先享受新的features

### Adobe Acrobat Pro DC（PDF编辑器）

> 个人认为重要程度大于Photoshop，很多时候你不能P图，但你需要处理PDF
>
> 破解版
> [百度云](https://pan.baidu.com/s/1gTELjmDeXPM6F_fn-a7X4g?pwd=8hmq)，更多信息见[文章开头，Adobe破解部分](#重大软院531实验室秘籍)

### KeePass2（密码存储、多端同步）

> 推荐和OneDrive一起使用，一般来说OneDrive有送的5G空间。直接把数据库放在OneDrive上，实现多端同步，非常简单

### Typora（Markdown编辑器）

> B站有破解教程，直接买也不贵

### Notepad++（代码文本编辑器）

> 什么都好就是作者政治倾向有问题，容易被恶心

### SumatraPDF（PDF浏览器）

> 可以和VSCode搭配使用，在本地搭建Texlive环境

### Bandizip（压缩软件）

### Overleaf (多用户协同，多版本)

> - 多用户协同是优于本地`Texlive+Git`，而且可以任意使用多版本`Texlive`，有时候需要切换版本
> - 注意Overleaf默认编译条件特别宽松，可以进行设定
> - Overleaf还可以直接发布，这个功能我还在探索中

### Rufus（ISO刻录软件）

> 好用，重装系统推荐，特别是Linux


## 如何找论文

> - Google Scholar 关键词，建议先找综述
> - 看最新顶会文章中的相关工作（很好用，一般来说**大组顶会**文章的相关工作做的非常详细）

### easyScholar（浏览器插件）

> 这个插件可以显示出论文发表期刊会议的CCF等级、所属出版商、IF、SCI分区等


## 如何用Latex 写 Pytorch pseudocode 

### 引入包

```latex
\usepackage[ruled,noline]{algorithm2e}
\usepackage{setspace}  % change the margin

\definecolor{commentcolor}{RGB}{110,154,155}   % define comment color
\newcommand{\PyComment}[1]{\ttfamily\textcolor{commentcolor}{\# #1}}  % add a "#" before the input text "#1"
\newcommand{\PyCode}[1]{\ttfamily\textcolor{black}{#1}} % \ttfamily is the code font
```

### 代码

```latex
\setlength{\algomargin}{0em}  % change the margin
\SetAlFnt{\small}             % set the font size
\begin{algorithm}[thb]
	\setstretch{0.8}          % chagne the line spacing
	\PyComment{this is a comment} \\
    \PyComment{this is a comment} \\
    \PyComment{} \\
    \PyComment{going to have indentation} \\
    \PyCode{for i in range(N):} \\
    \Indp   % start indent
        \PyComment{your comment} \\
        \PyCode{your code} \PyComment{inline comment} \\ 
        \Indm % end indent, must end with this, else all the below text will be indented
        \PyComment{this is a comment} \\
        \PyCode{your code}
    \caption{PyTorch-style pseudocode for PicT testing scheme}
    \label{algo:your-algo}
\end{algorithm}
```



## 如何在Latex中引入多个bib

[Creating multiple bibliographies in the same document - Overleaf, Online LaTeX Editor](https://www.overleaf.com/learn/latex/Questions/Creating_multiple_bibliographies_in_the_same_document#Packages_for_BibTeX)

### 使用multibib包

```latex
\usepackage[resetlabels,labeled]{multibib}

\newcites{Math}{Math Readings}
\newcites{Phys}{Physics Readings}
```

```latex
\cite{paper1} and \cite{paper2} were published later than 
\citeMath{paper3}. See also \citePhys{paper4}.

\bibliographystyle{unsrt}
\bibliography{references}

\bibliographystyleMath{unsrt}
\bibliographyMath{refs-etc}

\bibliographystylePhys{unsrt}
\bibliographyPhys{refs-etc}
```

> - 值得注意的是，使用该包会生成多个 .aux文件，必须对每个文件都进行bibtex编译后，再使用pdftex编译
> - 每个部分的bib文件最好单独存放，并单独命令各自的引用文章简写



## 如何找到论文复现代码（较为重要）

> - 先看论文本身，一般在**首页、摘要**有官方链接，或者**实验部分、补充文档**
> - **Github**，***直接在搜索框搜论文名字***，几乎所有放在github上的复现代码，都会在readme中写上论文标题。这种方法特别重要，因为一些实验室的代码可能并不够好或者用的Tensorflow，这种方法能找到一些大佬复现的Pytorch或者质量更高的代码
> - paperwithcode，这个网站初衷很好，但近几年几乎没有维护，个人感觉不好用



## 如何给服务器联网（dogcom）

> 现在外界流传的`dogcom`和`dogcom.conf`非常老，配置文件在个别服务器上出错，并且不能登出。我根据[drcom-generic/Drcom_CQU_HuxiCampus.py at master · drcoms/drcom-generic (github.com)](https://github.com/drcoms/drcom-generic/blob/master/custom/Drcom_CQU_HuxiCampus.py)重新写了一个python3的联网脚本，支持登出
>
> **Tips**：
>
> - 校园网使用哆点进行认证，github上有相关的仓库：[drcoms/drcom-generic: Dr.COM/DrCOM 现已覆盖 d p x三版。 (github.com)](https://github.com/drcoms/drcom-generic)、[mchome/dogcom at 309db8f545d7454b464a5d5d1d7dc4bde313f07a (github.com)](https://github.com/mchome/dogcom/tree/309db8f545d7454b464a5d5d1d7dc4bde313f07a)，如果有问题或者后续修改，可以查询相关仓库
>
> - `latest_w.py` 和 `dr.sh` 在 [/tools/dogcom/](https://github.com/DearCaat/CQU_531_Secrets/tree/main/tools/dogcom)下
>
> - 请修改`latest_w.py`文件中的34和35行 (`username`和`password`变量)，以使用自己的校园网账户进行登陆

```bash
# case1: 直接使用python文件登陆，这种情况会在终端显示log，并且随着终端的kill而kill，没有进程持久化
python3 latest_w.py
# case1: 按Ctrl+C，直接关闭该进程同时登出

# case2: 使用nohup维持进程持久化，我这里写了一个shell dr.sh
bash dr.sh
# case2: 这种情况下，请先使用ps找到你的python3进程
ps -aux | grep python3
# case2: 找到对应的 python3 latest_w.py 的进程号pid，kill该进程
kill -9 pid
# kill该进程的同时登出，相关的log会在./log下
```



## 如何在服务器上写代码（VSCode）

> VSCode有ssh插件，可以直接通过ssh连接服务器，并支持直接编辑服务器上的文件。扩展可以直接装在服务器上
>
> Tips: 
>
> - 可以使用公钥，避免每次打开文件都需要输入密码。（如果你已经在服务器`A`上配置了公钥登录，那只需要配置VSCode的ssh配置文件即可）简单流程如下：
>   - 本地主机创建公钥、私钥。公钥文件一般为`id_rsa.pub`，在`c:\Users\$username\.ssh\`下
>   - 复制你的公钥内容至服务器`A`的密钥认证文件`/home/$username/.ssh/authorized_keys`中
>   - 修改VSCode的ssh配置文件
> - 如果遇到ssh问题，请查看VSCode的`output`，一般来说将服务器上的`VSCode Server (/home/$username/.vscode-server/bin/$bin_code)`进行删除可以解决大部分问题。可以在`output`中查看到`$bin_code`



## 如何用更少显存更快更好跑实验 （AMP）

### tmux 使用

> 不要直接在ssh窗口开始跑实验，请使用`tmux`，其会创建一个持久化的窗口。以避免你当前窗口因特殊情况崩溃导致的实验中断

```shell
tmux new -s xxx
tmux a -t xxx
tmux kill-session -t xxx
```

### AMP使用

> 请最好使用AMP（半精度）训练，实验室GPU资源较为紧张，使用AMP训练可以有效降低显存，也不会损害你的性能。大厂都在用，一般来说训练速度还会有所提升

[Automatic Mixed Precision package - torch.amp — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/amp.html#autocasting)

### TIMM使用

> Backbone模型不推荐自己复现，也不推荐看CSDN复现，仅推荐查看相关领域论文或[timm](https://github.com/huggingface/pytorch-image-models)。NLP-Transformer类请参见[huggingface](https://github.com/huggingface/transformers)

### 训练代码

>训练代码不推荐完全自己写，也不推荐看CSDN，仅推荐查看相关领域论文（**大组**），或者大佬的，以下是我觉得很好的几个例子：
>
>[timm](https://github.com/huggingface/pytorch-image-models/blob/main/train.py)
>
>[swin](https://github.com/microsoft/Swin-Transformer/blob/main/main.py)
>
>- 推荐使用`yaml`文件存储config，而非全部config写在`parser`中
>- 两个例子中有特别多的`trick`，例如`batchsize scale, amp, ddp, gradient accumulate, gradient clip, mixup, checkpoint, ... `很多trick能有效提点，有些没有这么神奇



## 如何管理实验数据 （使用Wandb+Git）

> 管理实验数据、代码特别特别特别重要。一般来说实验代码会随着你的研究进程不断迭代，但一般人的水平都很难做到一直迭代还能保证能复现之前结果。因此最好对实验代码做版本管理
>
> **Wandb**：
>
> - 存储你的Git的`checkout`，本身也可以存储代码文件，详情查看文档
> - 记录所有`metrics`和`log`，并自动提供可视化，方便后续论文撰写
> - 注意，如果不设定存储路径（本地，在Pytorch代码中设定），会默认在你项目路径下备份一个实验数据（另一份传至服务器）。一般来说项目路径在`/home/$username/`下面，因此请设定存储路径到`/data/$username`
> - wandb也可以存储模型梯度，`wandb.watch()`，详情查看文档
> - 请不要建立太多`project`，因为实验室服务器只能有一个账号，所有人的`project`都在里面
> - 具体ip和账号名询问管理员

### 登录

[Basic Setup - Documentation (wandb.ai)](https://docs.wandb.ai/guides/self-hosted/local)

将`wandb.your-shared-local-host.com`替换成本地的`IP:Port`

```shell
wandb login --host=http://wandb.your-shared-local-host.com
```

在 http://wandb.your-shared-local-host.com/settings 界面可以找到***API keys***

> 下面部分是管理wandb私人服务器，如果只是使用请忽略。至于如何在代码中使用wandb，请查询官方document

### 重新创建App

不要对 ***Volumes*** 进行操作，这是保存数据的文件。可以删除 ***Containers*** ，然后在WSL中执行以下命令重新打开一个 ***APP***。 `--restart=always` 可以让容器每次随着docker服务启动

```shell
docker run -d --mount type=volume,src=wandb,dst=/vol -p 8080:8080 --name wandb-local --restart=always wandb/local
```

### 更新Image

```shell
docker pull wandb/local
```

### Volume的迁移

#### Backup Volume

目标volume：`wandb` ,保存位置：`/home/tangwenhao/twh` ，backup文件名：`backup.tar`

```shell
sudo docker run \
    --rm \
    -v wandb:/files_to_copy \
    -v /home/tangwenhao/twh:/place_to_paste \
    busybox \
    sh -c 'cd /files_to_copy && tar cf /place_to_paste/backup.tar .'
```

#### Restore Volume

目标volume：`wandb` ,保存位置：`/home/tangwenhao/twh` ，backup文件名：`backup.tar`

```shell
$ docker run \
    --rm \
    -v wandb:/place_to_paste \
    -v /home/tangwenhao/twh:/files_to_copy \
    busybox \
    sh -c 'cd /place_to_paste && tar xf /files_to_copy/backup.tar .'
```

### 记录一次登录失败处理

#### 情况：未知BUG，突然无法登录。

造成原因可能是 `$VOLUME_DIR/env/users.htpasswd` 下的记录消失，`user`表和`entities`表中仍有记录。无法登录，无法重置密码

#### 解决：重写入`/env/users.htpasswd`

利用之前备份的记录，重写入`/env/users.htpasswd`

```json
1070820773@qq.com:$2a$10$lupwoWa8YKrDmFqKdu4huOqAVHFdCGjHE/ZZG8QpPXRTa.jdPSuWq
#对应密码：123456789
```

恢复正常

### 记录一次更新lincese的处理

[system-admin -- Application Error Unable to reach the backend api. If this perists, check the system logs. · Issue #46 · wandb/server (github.com)](https://github.com/wandb/server/issues/46)



## 如何远程操作实验室电脑

### Windows远程桌面连接（只适用于内网）

> 比向日葵好用100倍，但**只适用于内网**，但需要**Windows专业版**（在你需要被远程的那个主机上安装，淘宝10元）

### 向日葵



## 如何在多个服务器之间同步代码

### Git

> 选择其中一个服务器`A`作为远程仓库，保证其它所有服务器都能与`A`进行ssh连接。这里推荐ssh公钥的形式

- 这里需要在其余服务器`B,C,D,...`上初始化公钥、私钥（如果已经有请跳过，一般来说存放在`/home/$username/.ssh/`下名字为`id_rsa.pub`）

- 将服务器`B,C,D,...`上的公钥中的内容复制下来，粘贴到服务器`A`的认证密钥文件（`/home/$username/.ssh/authorized_keys`）中。

- 在服务器`A`上初始化`git服务器`其等效于远程仓库，获得目录`/home/$username/.../xx.git`

  ```shell
  git init --bare xx.git
  ```

- 在服务器`B,C,D,...`上添加远程仓库，通过ssh的形式

  ```shell
  ssh://$username@Server_A_IP:/home/$username/.../xx.git
  ```

  这里的`Server_A_IP`是服务器`A`的IP。

- **Tips**：就算在服务器`A`上，也要在另外一个目录放置代码，然后使用`git`进行代码管理，不能直接修改服务器`A`上的`git服务器`的代码。

### SyncTrayzor

GitHub repo: https://github.com/canton7/SyncTrayzor



## 如何抢卡（慎用慎用慎用）

> 最好不要用，但紧急情况下，可以私下联系我。后续更新



## 如何重装Ubuntu

### 保留/home

> 主要难点在于之前的系统引导（UEFI或者Legacy）要与当前对应。不然在新系统分区那里，没办法在原硬盘上设定安装引导程序
>
> **重点在于一定要备份！！！**

### Tips

> - 没网的时候注意不要先设定网络连接，会导致后面安装失败
> - 禁用Sleep：`systemcyl status sleep.target` 可以查看是否开启该服务。禁用：`sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target`

### 设置网络环境（设置IP）

> - 一般Ubuntu使用`/etc/netplan` 或者 `/etc/nework` 配置IP，这俩只能取其一。配置文件就在其中。
> - 要弄明白`dhcp`和`static`，机房默认是`static`，找管理员要网关、DNS
> - 记得备份默认的配置文件 `mv default.yaml default.yaml.backup`
> - 虽然这两个工具都可以热重启，但还是推荐重启服务器，可以省很多麻烦
> - 注意网线插的网口是否和配置文件一致，如果一直不起效，请将网线插入其它网口试试

### 安装NVIDIA驱动

#### 禁用nouveau驱动

编辑 `/etc/modprobe.d/blacklist-nouveau.conf` 文件，添加以下内容：

```shell
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
```

关闭nouveau：

```shell
echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
```

重新生成内核并重启：

```shell
sudo update-initramfs -u
sudo reboot
```

重启后，执行：`lsmod | grep nouveau`。**如果没有屏幕输出，说明禁用nouveau成功。**

#### Tips

> - 注意，安装的driver版本最好和CUDA版本相对应。各个CUDA版本的Doc：[CUDA Toolkit Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive)。在此处查看：[Release Notes :: CUDA Toolkit Documentation (nvidia.com)](https://docs.nvidia.com/cuda/archive/11.7.0/cuda-toolkit-release-notes/index.html)
> - `nvidia-smi` 有时候很慢，注意更改持久模式：`sudo nvidia-persistenced --persistence-mode` 或者 `sudo nvidia-smi -pm 1`
> - Ubuntu重启有可能找不到Driver，一般情况是内核进行了更新。使用`ls /usr/src`查看驱动版本号，然后生成导向模块`sudo dkms install -m nvidia(nvidia-srv) -v 530.30.02`。[Ubuntu20.04重启后找不到Nvidia显卡驱动 - keep-minding - 博客园 (cnblogs.com)](https://www.cnblogs.com/minding/p/17449134.html)

#### 安装Driver

采用apt方法

使用命令`ubuntu-drivers devices`获取可用驱动信息，如果命令不存在自己安装一下。

`sudo apt install nvidia-driver-***-server`

#### 安装CUDA

[CUDA Toolkit Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive)

选择对应的版本，最好采用runfile的安装形式，有一个交互过程，**deb的安装会默认替换掉你先有的driver，非常麻烦**

`sudo sh cuda_11.4.0_470.42.01_linux.run`

**切记：安装driver之后，不要在安装CUDA时再次安装！！！**

**注意，当提醒你已经安装了driver的时候，直接continue。当选择安装内容的时候，务必把driver前面的x取消掉，因为我们已经安装了驱动！！！**

添加环境变量：

`vim /etc/profile` ：（将 `***` 替换为对应的CUDA版本）

```shell
export PATH=/usr/local/cuda-***/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-***/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

`nvcc -V` 显示相关信息，表示安装成功

#### 安装CUDNN

[CUDA Deep Neural Network (cuDNN) | NVIDIA Developer](https://developer.nvidia.com/cudnn)

[Installation Guide :: NVIDIA Deep Learning cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

最好下载TAR，然后复制

解压：

```shell
tar -xvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz
```

复制：

```shell
$ sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
$ sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
$ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```



## 如何在RootLess的情况下使用Docker

### 将普通用户添加到Docker用户组

[docker命令 报Got permission denied while trying to connect to 错误 - ranh - 博客园 (cnblogs.com)](https://www.cnblogs.com/hwwei/p/16018100.html)

```shell
sudo usermod -aG docker $username
```

### 使用Root-less Docker

> Rootless mode allows running the Docker daemon and containers as a non-root user to mitigate potential vulnerabilities in the daemon and the container runtime.
>
> Rootless mode does not require root privileges even during the installation of the Docker daemon, as long as the [prerequisites](https://docs.docker.com/engine/security/rootless/#prerequisites) are met.
>
> Rootless mode was introduced in Docker Engine v19.03 as an experimental feature. Rootless mode graduated from experimental in Docker Engine v20.10.

[Run the Docker daemon as a non-root user (Rootless mode) | Docker Documentation](https://docs.docker.com/engine/security/rootless/)

***这种情况下，系统中有两个单独的Docker，相互独立***

***rootless Docker存在一定限制：***

[Run the Docker daemon as a non-root user (Rootless mode) | Docker Documentation](https://docs.docker.com/engine/security/rootless/#known-limitations)

**刷新配置，重启服务：**

```shell
systemctl --user daemon-reload
systemctl --user restart docker
```

更多Rootless的 [TIPS](https://blog.pulipuli.info/2023/03/blog-post_19.html)

#### **Ubuntu18.04有坑，按照官方教程无法正常安装**

[Dockerd-rootless.sh not working on Ubuntu 18.04 because a dependency is missing (vpnkit or slirp4netns) · Issue #41781 · moby/moby (github.com)](https://github.com/moby/moby/issues/41781)

- 需要手动安装[`slirp4netns`包]([Releases · rootless-containers/slirp4netns (github.com)](https://github.com/rootless-containers/slirp4netns/releases))，这个包无法通过apt-get安装

### 配置仓库镜像

`daemon.json` 文件应该在 `~./config/docker/` 下

***绝大多数mirror都无法正常使用，阿里云的已经不再更新，`dockerproxy.com`好像还行，不知道是个啥组织***

***学校的网好像可以直接访问官方hub***

## 如何将Keras模型 (Tensorflow) 转Torch

[AgCl-LHY/Weights_Keras_2_Pytorch: a python code to convert Keras pre-trained weights to Pytorch version (github.com)](https://github.com/AgCl-LHY/Weights_Keras_2_Pytorch)

***关键在于Keras一些层权重包的存储方式和Torch不同，要进行一个维度转化***

```python
def keras_to_pyt(km, pm):
    weight_dict = dict()
    for layer in km.layers:
        if type(layer) is keras.layers.convolutional.Conv2D:
            if (len(layer.get_weights()) >= 1):
                weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (3, 2, 0, 1))
            if (len(layer.get_weights()) >= 2):
                weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
        elif type(layer) is keras.layers.Dense:
            if (len(layer.get_weights()) >= 1):
                weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (1, 0))
            if (len(layer.get_weights()) >= 2):
                weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
        elif type(layer) is keras.layers.DepthwiseConv2D:
            if (len(layer.get_weights()) >= 1):
                weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (2, 3, 0, 1))
            if (len(layer.get_weights()) >= 2):
                weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
        elif type(layer) is keras.layers.BatchNormalization:
            if (len(layer.get_weights()) >= 1):
                weight_dict[layer.get_config()['name'] + '.weight'] = layer.get_weights()[0]
            if (len(layer.get_weights()) >= 2):
                weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
            if (len(layer.get_weights()) >= 3):
                weight_dict[layer.get_config()['name'] + '.running_mean'] = layer.get_weights()[2]
            if (len(layer.get_weights()) >= 4):
                weight_dict[layer.get_config()['name'] + '.running_var'] = layer.get_weights()[3]
        elif type(layer) is keras.layers.ReLU:
            pass
        elif type(layer) is keras.layers.Dropout:
            pass
```



## 如何仅让部分IP通过指定VPN（Windows  默认VPN 修改路由）

[Automatically Add Static Routes After Connecting to VPN | Windows OS Hub (woshub.com)](http://woshub.com/add-routes-after-connect-vpn-windows/)

![uncheck the option "Use default gateway in remote network" in the vpn connection ipv4 properties](http://woshub.com/wp-content/uploads/2021/09/uncheck-the-option-use-default-gateway-in-remote.png)

OR 

```shell
Set-VpnConnection –Name workVPN -SplitTunneling $True
```

添加路由：

```shell
Add-VpnConnectionRoute -ConnectionName workVPN -DestinationPrefix 192.168.11.2/32 -PassThru
```

该指令是在你激活VPN的时候添加路由信息，在退出VPN服务器后，删除该路由

删除路由：

```shell
Remove-VpnConnectionRoute -ConnectionName workVPN -DestinationPrefix 192.168.111.0/24 -PassThru
```

## 如何传输数据集

直接使用xftp进行拖拽

### 对于数目比较多的图片文件夹，可以使用Sftp进行传输

* 登录: sftp -P port user@ip  (-P 是大写)
* 使用 ls 命令列出目录，使用 "cd CloudData" 命令进入数据根目录
* 使用 "get <文件名>", 从云盘下载文件到本地当前目录
* 使用 "get -r <文件夹名>", 从云盘下载目录到本地当前目录
* 使用 "put <文件名>", 把当前目录的本地文件上传到云盘
* 使用 "put  -r <文件夹名>", 把本地当前目录上传

## Docker 支持CUDA

### 安装[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#docker)

***如果你是Root-less，有坑！！***

- 下面命令会报错，因为Rootless的配置文件位置不同

```shell
sudo nvidia-ctk runtime configure --runtime=docker
```

这个时候建议你在普通Docker配置文件应在位置`/etc/docker`建一个文件，然后复制改写后的内容到Rootless的配置文件`.config/docker`下。

- 还要记得改` /etc/nvidia-container-runtime/config.toml`，将里 面的`no-cgroups`设为`true`

**Tips:**

- 宿主主机的Driver版本跟你在Container中能用的CUDA版本挂钩，你Driver版本太低用不了高版本的CUDA。其中`495（cuda 11.5）和520 (cuda 11.8)`版本的driver几乎没有兼容性，宿主主机千万别是这俩。[CUDA兼容性问题（显卡驱动、docker内CUDA） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/459431437)
- CUDA的Forward compatibility好像仅对Tesla架构的显卡有用，所以最好升级你的driver。[PyTorch的CUDA错误：Error 804: forward compatibility was attempted on non supported HW - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/361545761)

### 下载或者配置Image

DockerHub 中 Torch和Nvidia官方的image都有`devel`这个版本，大小会比普通的`runtime`大很多。如果你需要在image中使用cuda编译某个包，就需要`devel`，如果你只是想用下`pytorch-gpu`那就完全没必要。nvidia的官方base-image还要选择是否使用`cudnn`

## 在线升级Ubuntu服务器

### 备份

`sudo timeshift --help`

### 升级

[服务端升级Ubuntu 20.04 LTS 记录 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/136109436)

- 不要无脑continue，一定要看清楚，保证网络畅通，记得提前更换apt源，不然肯定失败

- 除了sshd的配置之外，其它配置都推荐不要维持原版本，sshd不知道也不会不会冲突，最后都用新的自己重新配置

## Ubuntu配置Clash

[Releases · Dreamacro/clash (github.com)](https://github.com/Dreamacro/clash/releases)

需要手动下载[Country.mmdb](https://link.zhihu.com/?target=https%3A//gitee.com/mirrors/Pingtunnel/blob/master/GeoLite2-Country.mmdb)，将其改名为`Country.mmdb`

下载`proxychains4`，配置目录`/etc/proxychains.conf`

- 改为 `dynamic_chain`
- 添加代理ip，好像不能添加`https`

测试是否成功

```shell
proxychains4 curl www.httpbin.org/ip
```

**非root可能报错**，见[proxychains4配置使用 - 0xcreed - 博客园 (cnblogs.com)](https://www.cnblogs.com/mwq1024/p/11582003.html)

**proxychains4 对docker无效，docker代理需要单独配置**

## Ubuntu下挂载移动硬盘并修改权限

- 一次性，指定`uid`可以正常使用用户读写文件夹，不然权限错误

  ```shell
  sudo mount -o uid=$UID /dev/sdc1 /mnt/book
  ```

- 每次开机自动挂载

  [ubuntu自动挂载硬盘实现所有用户可读写 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/163001267)

  ```shell
  # 查看 TYPE 和 UUID
  sudo blkid
  # 编辑 /etc/fstab文件
  UUID=$UUID /mnt/book $TYPE defaults,uid=$UID,gid=$GID 0 0
  ```

## 跨网段ssh连接

> 该问题核心在于内网穿透，但**ipv6**解决了该问题，因此下面都是基于通信两边都具有公网ipv6且已经解决了路由器问题和防火墙问题
>
> sftp: 要注意加上`-6`，不然默认会解析成ipv4的地址，还要把地址加上`[]`
>
> ssh: 直接使用ipv6地址即可

## 搭建私有Docker Registry

~~[使用docker-compose搭建私有docker registry - 落叶&不随风 - 博客园 (cnblogs.com)](https://www.cnblogs.com/xpengp/p/12714381.html)~~

[Docker Compose 部署配置和使用 Registry 私有镜像仓库 - 思有云 - IOIOX](https://www.ioiox.com/archives/140.html)

> TODO:
>
> - [ ] 令牌认证，对不同用户的权限进行规范

- 启动一个一次性容器用于创建账号密码.密码文件路径以`/root/registry/htpasswd`为例,账号密码以`admin`和`12345678`为例.

  ```shell
  docker run --rm --entrypoint \
      htpasswd httpd:2 -Bbn \
      admin 12345678 > /root/registry/htpasswd
  ```

- `daemon.json`写入：

  ```json
  {
      "insecure-registries": [
          "[私有仓库 ip:port]"
      ]
  }
  ```

- `docker-compose.yaml`，该文件可以在任意位置，后续`-f`指定即可

  ```yaml
  version: '3'
  
  services:
    registry:
      image: registry:2
      restart: "always"
      ports:
        - 5000:5000
      environment:
        - REGISTRY_AUTH=htpasswd
        - REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd
        - REGISTRY_AUTH_HTPASSWD_REALM=Registry Realm
        - REGISTRY_STORAGE_DELETE_ENABLED=true
      volumes:
        - $DATA_ROOT:/var/lib/registry
        - $PASSWD_ROOT:/auth/htpasswd
  ```


- 使用`docker-compose`部署仓库

  ```shell
  docker-compose -f $DOCKER_COMPOSE_CONFIG up -d
  ```


### 删除Image

[HTTP API V2 | Docker Documentation](https://docs.docker.com/registry/spec/api/#deleting-an-image)

[Docker私有仓库Registry删除镜像的方法【20220321】 - 鬼谷子叔叔 - 个人主页 (tongfu.net)](https://tongfu.net/home/35/blog/513697.html)

[docker私有镜像仓库搭建和镜像删除 - 简书 (jianshu.com)](https://www.jianshu.com/p/b93feaf43f37)

这个比较麻烦，总的来说，可以通过API的方法来删，但也需要修改container里面的config，位置在`/etc/docker/registry/config.yml`，然后查询`digest`，然后根据`digest`调用`DELETE` API进行删除，最后垃圾回收

第二个攻略说是没有删干净，但我发现其实主体文件通过以上步骤可以完全删除，只是会留下一个registry，占用空间很小

## 用Docker跑实验

### Image管理

先看本地有什么现存的`image`

```shell
docker image ls
```

跑实验需要的特制化`image`都放在本地仓库中，通用的请查看 [Docker_Hub](https://hub.docker.com/)。先查看本地仓库有哪些`image`,本地仓库`ip`默认为`10.236.11.202:5000`

```shell
# 登陆本地仓库，输入账户名和密码
docker login $LOCAL_REGISTRY_IP

# 先查看有哪些image
curl -u $USER:$PASSWD http://$LOCAL_REGISTRY_IP/v2/_catalog

# 再查看具体的image下有哪些tag
curl -u $USER:$PASSWD http://$LOCAL_REGISTRY_IP/$IMAGE_NAME/tags/list
```

`pull, push`,***非特殊情况，不要随意push image到本地仓库中，就算要push，也要在commit的时候做好comment，然后 tag写清楚。目前本地仓库没有做权限限制，登陆后就可以对上面的所有image做任何操作***

```shell
# PULL跟常规一样，只是要注意要在image前面加上本地仓库的Ip，否则会默认从docker_hub获取
docker pull $LOCAL_REGISTRY_IP/$IMAGE_NAME:$TAG

# push
# push之前需要先commit一个本地的image
docker commit -a $AUTHOR_NAME -m $COMMENT $CONTAINER $LOCAL_REGISTRY_IP/$IMAGE_NAME:$TAG
docker push $LOCAL_REGISTRY_IP/$IMAGE_NAME:$TAG
```

### 跑实验

先看是否创建了`container`:

```shell
docker ps -a
```

docker 管理工具推荐：portainer, 可以方便的删除，重启，暂停，重新配置docker容器。设定好docker容器的配置文件夹，可以很方便的迁移。
```shell
docker pull portainer/portainer
docker run -d -p 9000:9000 -v /var/run/docker.sock:/var/run/docker.sock --restart=always --name prtainer portainer/portaine
```


如果已经有创建好的container就不要再`run`

```shell
# 针对已经stopped的container
docker start -a $CONTAINER_NAME

# 针对还在运行中的container
docker attach $CONTAINER_NAME
```

如果没有现存`container`，就直接从`run image`。要注意的是，`timm`库和`torch`的模型默认的权重文件存放在`./cache/huggingface`和`./cache/torch`中，最好也把这两个文件夹做映射避免在不同`image`中重复下载。尽量只映射`huggingface`和`torch`这两个子文件夹，`$LIB`就是`huggingface`或者`torch`。

```shell
docker run --gpus all -it --shm-size 32g -p $CONTAINER_SSH_PORT:22 -v $CODE_DIR:/workspace/code -v $DATASET_DIR:/workspace/dataset -v $OUTPUT_DIR:/workspace/output -v $CACHE_DIR:/root/.cache/$LIB --name $CONTAINER_NAME $IMAGE_NAME
```

### SSH连接

想要使用`ssh`连接`container`

- 首先要在创建时进行端口映射`-p $CONTAINER_SSH_PORT:22` 

- 将客户端公钥放入`/root/.ssh/authorized_keys`，可能需要创建该文件
- 记得查看`container`中是否已经打开`ssh_server`服务：`service ssh start` 
- 建议使用`tmux`进行实验，确保在`terminal`崩溃的情况下，实验能够继续
  
### 网络相关

- 最好直接使用`host`网络（`--network host`），跑实验用的`container`个人感觉不需要复杂的通讯，其带来的好处和便利非常诱人，例如：完整的`ipv6`支持

### 重装驱动

- 首先查看当前驱动版本
```shell
ls /usr/src | grep nvidia
```
- 重新安装现有的驱动版本
```shell
sudo dkms install -m nvidia -v 450.57 # 版本即使上一个命令的输出
# 如果没有安装dkms包
sudo apt-get install dkms
```
- 重启
```shell
sudo reboot
```
