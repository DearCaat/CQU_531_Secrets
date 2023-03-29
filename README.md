# 重大软院531实验室秘籍
我发现在研究生学习中遇到的很多问题都是相似的，不管是科研任务、代码实现、服务器管理。但一届一届的师兄师弟都在找着同样的资料，浪费着同样的时间。我希望这种经验贴能够帮助新老同学，新同学能够更快入门，老同学没必要去纠结自己一年前解决过的问题。个人能力有限，希望同学们一起分享、订正、补充 🤗。

**Notes:** 很多内容没有具体详细的代码和指令，只会提供一个思路或者关键词，希望同学们能够善用**ChatGPT**、**Bing AI**、**Google** 😀。

**Status:** 整理中~~~~~~~



## Table of Contents

- [论文]()
  - [如何找论文](#如何找论文)
  - [如何写Pytoch格式的伪代码](#如何用latex-写-pytorch-pseudocode)
  - [如何插入多个参考文献小节](#如何在latex中引入多个bib)
- [实验](#)
  - [如何用更少显存更快更好跑实验 （AMP）](#如何用更少显存更快更好跑实验-amp)
  - [如何管理实验数据（使用Wandb+Git）](#如何管理实验数据-使用wandbgit)
  - [如何管理wandb服务器](#重新创建app)
  - [如何远程操作实验室电脑](#如何远程操作实验室电脑)
  - [如何重装Ubuntu](#如何重装ubuntu)
  - [如何装Nvidia3件套](#安装nvidia驱动)
  - [如何在多个服务器之间同步代码](#如何在多个服务器之间同步代码)
  - [如何在Rootless的情况下使用Docker](#如何在rootless的情况下使用docker)
  - [如何将Tensorflow模型转Pytorch](#如何将keras模型-tensorflow-转torch)
  - [如何仅让部分IP通过指定VPN](#如何仅让部分ip通过指定vpnwindows--默认vpn-修改路由)

- [工具](#工具推荐)



## 工具推荐

工欲善其事必先利其器

### ReadPaper（论文管理）

> 我尝试使用过Zotero这种自定义更高的工具，但**ReadPaper**相关功能更易用且更完善。个人推荐。

### Edge浏览器（多端同步）

> 可以在设置里面选择下载预览版（`dev`版），一般来说可以先享受新的features

### Adobe Acrobat Pro DC（PDF编辑器）

> 个人认为重要程度大于Photoshop，很多时候你不能P图，但你需要处理PDF
>
> 破解版有需要联系学长

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

## 如何找论文
> - Google Scholar 关键词，建议先找综述
> - 看最新顶会文章中的相关工作（很好用，一般来说**大组顶会**文章的相关工作做的非常详细）

### easyScholar（浏览器插件）

> 这个插件可以显示出论文发表期刊会议的CCF等级、所属出版商、IF、SCI分区等

## 如何在多个服务器之间同步代码

### Git

> 选择其中一个服务器`A`作为远程仓库，保证其它所有服务器都能与`A`进行ssh连接。这里推荐ssh公钥的形式

- 这里需要在其余服务器`B,C,D,...`上初始化公钥、私钥（如果已经有请跳过，一般来说存放在`/home/$username/.ssh/`下名字为`id_rsa.pub`）

- 将服务器`B,C,D,...`上的公钥中的内容复制下来，粘贴到服务器`A`的认证密钥文件（`/home/$username/.ssh/authorized_keys`）中。

- 在服务器`A`上初始化`git服务器`其等效于远程仓库，获得目录`/home/$username/.../xx.git`

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
sudo groupadd docker # 添加docker用户组
sudo gpasswd -a $USER docker #将登录用户加入到docker用户组中
newgrep docker #更新用户组
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

### 配置仓库镜像

`daemon.json` 文件应该在 `~./config/docker/` 下



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

