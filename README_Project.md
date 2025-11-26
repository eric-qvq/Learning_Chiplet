# Learning_Chiplet
前言：参加了这个社区比赛，感觉能学习到一些知识，打算记录下自己写的东西。

集成芯片和芯粒技术开源社区大赛：面向多芯粒异构集成的体系结构级仿真器搭建与设计拓展
---
## 赛题描述
阶段一：完成仿真环境搭建与基准测试运行 

阶段二：修复现有Benchmark 并设计新测试用例 

阶段三：集成自定义CPU模拟器并开展对比实验

阶段四：分析互连架构性能并提出优化方案

### 阶段一：按照说明安装并运行仿真器成功 

#### 1.1 赛题要求 

1. 按照说明文档安装仿真器（但是里面没有环境安装具体的指令，特别是GPGPU-sim 需要的 CUDA 安装，需要自行百度）

#### 1.2 赛题目标 

1. 运行自带的benchmark 成功，运行界面无报错，每个阶段每个芯粒的log 文件有对应内容

对于上述的要求和目标

先安装好各自的依赖包 `gem5` , `gpgpu-sim` ,  `snipersim` （snipersim可以在后面安装的时候根据提示来安装）

1.从github上下载仓库。

```
git clone --single-branch --branch master_v2 https://github.com/FCAS-LAB/Chiplet_Heterogeneous_newVersion.git
```

进入仿真器根目录，以下的示例命名都假设从仿真器根目录开始执行。

2.初始化并更新submodule。

```git submodule init
git submodule init
git submodule update
```

3.运行脚本，初始化环境变量

``` 
source setup_env.sh 
```

运行成功应出现：setup_environment succeeded

> [!NOTE]
>
> 如果出现了`GPGPU-Sim version 4.0.0 (build gpgpu-sim_git-commit-a4ce3fe-modified_0.0) ERROR ** Install CUDA Toolkit and set CUDA_INSTALL_PATH.` 
>
> 那就是没有安装CUDA，需要安装CUDA才能让gpgpu-sim配置环境成功
>
> 由于gpgpu-sim的版本是4.0.0版本，所以按照CUDA的版本不能高于CUDA 11 的版本，需要注意这个点

安装完CUDA之后会出现：setup_environment succeeded



4.对于snipersim和gpgpu-sim代码进行修改。

```
1.patch.sh脚本用来生成Patch
./patch.sh    

2.apply_patch.sh脚本用来应用Patch
./apply_patch.sh
```

打包和应用Patch

由于sniper和GPGPUSim是用submodule方式引入的，对于snipersim和gpgpu-sim的修改不会通过常规的git流程追踪。因此，工程提供了patch.sh和apply_patch.sh两个脚本通过Patch管理sniper和gpgpu-sim的修改。



patch.sh脚本用来生成Patch：

```
./patch.sh
```

1. 使用patch.sh脚本将snipersim和gpgpu-sim的修改分别打包到snipersim.diff和gpgpu-sim.diff文件中。diff文件保存在interchiplet/patch下面。diff文件会被git追踪。
2. patch.sh脚本还会将被修改的文件按照文件层次结构保存到.changed_files文件夹中，用于在diff文件出错时进行查看和参考。



apply_patch.sh脚本用来应用Patch：

```
./apply_patch.sh
```

1. 使用apply_patch.sh脚本将snipersim.diff和gpgpu-sim.diff文件应用到snipersim和gpgpu-sim，重现对于文件的修改。
2. 当apply出错时，可以参考.changed_files中的文件手动修改snipersim和gpgpu-sim的文件。

需要说明的是：不建议用.changed_files直接覆盖snipersim和gpgpu-sim文件夹。因为snipersim和gpgpu-sim本身的演进可能会与芯粒仿真器修改相同的文件。使用Patch的方式会报告修改的冲突。如果直接覆盖，则会导致不可预见的错误。



5.编译安装snipersim。新版本的snipersim提供了非常自动化的编译脚本，直接执行make即可。

```
cd snipersim
make -j4
```

> [!NOTE]
>
> 在编译`snipersim`的时候会出现需要安装的依赖包，安装它给的提示来安装即可编译，之后按照文档链接进去，按照里面的要求安装下载，然后设置环境

之后去测试snipersim能否运行，然后cd到 test/fft 运行make run 发现是能够成功运行显示详细模拟结果与性能统计数据



6.编译安装Gem5。请查看Gem5文档获取详细安装指南。LegoSim中可以运行X86和ARM架构仿真器

这里可以通过指令`uname -m`查看自己的虚拟机是什么类型的架构 ，然后对应的去编译gem5比较好一点

```
cd gem5
scons build/X86/gem5.opt
```

或者

```
cd gem5
scons build/ARM/gem5.opt
```
Gem5编译完能成功生成opt文件


7.编译安装GPGPUSim。GPGPUsim安装有前置条件：

1. GPGPUSim需要安装cuda。新版本的gpgpusim可以支持cuda4到cuda11的任意版本，详细信息请参见GPGPUSim的README。
2. GPGPUSim对于编译版本有要求，建议使用GCC7 

配置CUDA和GPGPUSim的Dependencies(依赖项)

GPGPU-Sim dependencies:

```
sudo apt-get install build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev
```

GPGPU-Sim documentation dependencies:

```
sudo apt-get install doxygen graphviz
```

AerialVision dependencies:

```
sudo add-apt-repository ppa:linuxuprising/libpng12
sudo apt update
sudo apt install libpng12-0
sudo apt-get install python-pmw python3-ply python3-numpy python3-matplotlib
```

CUDA SDK dependencies:

```
sudo apt-get install freeglut3-dev
sudo apt-get install libxi-dev libxmu-dev 
```

配置好后，可以去安装CUDA了

配置好CUDA和编译器，可以直接执行make。

```
cd gpgpu-sim
make -j4
```


> [!NOTE]
>
> 如果要重新make GPGPU-Sim 需要先把环境配置下,部署GPGPU-Sim，在make
>
> ```
> source setup_environment
> ```
>

8.编译安装popnet

```
cd popnet_chiplet
mkdir build
cd build
cmake ..
make -j4
```

9.编译安装芯粒间通信程序。interchiplet提供了芯粒间通信所需要的API和实现代码。

```
cd interchiplet
mkdir build
cd build
cmake ..
make
```


编译完成后应在interchiplet/bin下找到record_transfer和zmq_pro，在interchiplet/lib下找到libinterchiplet_app.a。

> [!NOTE]
>
> 在interchiplet目录下的CMakeLists.txt中是没有record_transfer.cpp和zmq_pro.cpp相关内容的，同时AI了下，了解了作者应该是没有把这两个.cpp放进去，然后是放在了deprecated文件夹里面，deprecated意为“已弃用”，这也表面了`zmq_pro` 和 `record_transfer` 这两个程序已经被新的实现所取代。同时在deprecated文件夹里面也有CMakeLists.txt，然后里面是有`zmq_pro` 和 `record_transfer` 这两个程序的，因此，可以得知编译得到的 `../bin/interchiplet` 很可能就是这两个旧程序的**替代品**，是一个功能更全或架构更新的统一可执行文件。而 `../lib/libinterchiplet_app.a` 是编译这个新程序（以及可能其他应用）时所依赖的库。
>
> 所以在interchiplet/bin下是找不到record_transfer和zmq_pro的编译后的文件，应该是新的文件，只是作者可能没有更新这个README文档导致的。



上述安装好了之后可以进行验证了

正确执行上述过程后，可以使用benchmark/matmul验证环境设置是否正确。

1.设置仿真器环境

```
source setup_env.sh
```


2.编译可执行文件

```
cd benchmark/matmul
make
```


3.执行可执行文件。示例包含4个进程，分别是1个CPU进行和3个GPU进程。必须在benchmark/matmul进程执行。

```
../../interchiplet/bin/interchiplet ./matmul.yml
```

执行后，可以在benchmark/matmul文件下找到一组proc_r{R}_p{P}_t{T}的文件夹，对应于第R轮执行的第P阶段的第T个线程。 在文件夹中可以找到下列文件：

1. GPGPUSim仿真的临时文件和日志文件gpgpusim_X_X.log。
2. Sniper仿真的临时文件和sniper仿真的日志文件sniper.log。
3. Popnet的日志文件popnet.log。


4.清理可执行文件和输出文件。

```
make clean
```

---


### 阶段二：使用仿真器运行benchmark 

#### 2.1 赛题要求 

1. 本项目benchmark 中的 mlp 构建有 bug，会陷入死循环，请找到并修改这个bug；
2. 请构建新的benchmark，引入新的条件变量，探索本仿真器的使用边界。 

#### 2.2 赛题目标 

1. 运行 mlp 成功，运行界面无报错，每个阶段每个芯粒的log文件有对应内容；
2. 构建新的 benchmark，能够成功运行，且运行界面无报错，每个阶段每个芯粒的 log 文件有对应内容。


要求一，里面的构建有bug是指的当CPU跟GPU进行数据传输，在最后CPU把输出都传完了，GPU按理来说应该是结束了，但因为里面的while(1)，导致一直没结束，一直在循环空转中，等待CPU传下一个根本不存在的指令。对于这部分的修改bug，需要在GPU和CPU的通信过程中，加入终止信号，让GPU能够退出循环，从而进程终止。


要求二，构建了新的边界基准：CPU 驱动程序现在发送每个芯片的元数据（长度 + 限制），流式传输可变大小的 int32 数据负载，并验证 GPU 结果。每个目标使用不同的消息大小来探测 NoC 上的边界条件。实现了 GPU 工作程序，它接收元数据/负载，限制值，执行块级归约，记录工作，并返回总和。添加了构建/运行管道：Makefile 用于生成 bin/boundary_cpu 和 bin/boundary_cu，以及 boundary.yml，其中包含 2×2 网格设置（GPU 位于 (0，1)、(1，0)、(1，1)，CPU 位于 (0，0)）和配置为相同网格的 PopNet 阶段。

当前可调的关键参数及默认值如下：


CPU (boundary.cpp)

1. 目标列表 targets（目的坐标 + 负载长度 + clamp 值
2. 随机数分布：std::uniform_int_distribution<int32_t> dist(-1500, 1500)（控制负载数据范围）。
3. 发送顺序：先 16 字节元数据（len, clamp），再 payload，再接收 int64 结果。


GPU (boundary.cu)

1. 线程块配置：threads = 256；block 数 blocks = (h_len + threads - 1) / threads。
2. 共享内存大小：shared_bytes = threads * sizeof(int64_t)。
3. clamp_and_sum 核心使用 clamp 限制值、长度 h_len。
4. 交互顺序：接收 16 字节元数据 → 接收 payload（sizeof(int32_t)*h_len）→ 归约 → 发送 int64 结果。

芯粒布置 (boundary.yml, Phase2)

PopNet/拓扑

1. 网格与通道
2. 缓冲
3. 链路/时间
4. 路由

可以调整以上参数（例如增大/减小 len、clamp，改变随机分布范围，改线程数、缓冲大小、虚通道数、链路长度、仿真周期等）来探索边界条件。


---

### 阶段三：设计拓展

#### 3.1 赛题要求 

1. 在现有仿真器基础上，参考下面的链接：https://github.com/FCAS-ZJU/Chiplet_ Heterogeneous_newVersion/wiki/Imported-Simulator 引入新的 CPU 仿真器。这个仿真器可以自己设计，也可以参考现有的开源项目进行修改；这个模拟器可以设计的很简单，只要满足基本的功能需求即可。具体案例参考附录。 
2. 在此仿真器上运行mlp benchmark。 


#### 3.2 赛题目标 

1. 成功设计或引入新的CPU仿真器，实现任意benchmark的运行； 
2. 成功运行mlp，比较其与现有仿真器的性能差异。

任务一采用参考现有的开源项目进行修改，因为感觉用其他的比较麻烦，所以采用他自带的gem5来代替sniper去做CPU仿真器的效果


任务二对于运行mlp的要求目标暂时做不了。

---

### 阶段四：芯粒互联架构分析与应用

#### 4.1 赛题要求 

1. 分析仿真器中芯粒互联机制的工作原理；
2. 构建专用benchmark 测试芯粒互联性能； 
3. 提出至少一种改进芯粒互联模块的建议。 


#### 4.2 赛题目标 

1. 成功运行芯粒互联benchmark，记录测试结果； 
2. 实施改进建议并验证性能提升。



要求1.分析仿真器中芯粒互联机制的工作原理

拓扑与路由机制分析：

体现：在 cic.yml 中，构建了一个 2x2 的网格系统（Mesh），包含 1 个 CPU (0,0) 和 3 个 GPU ((0,1), (1,0), (1,1))。

这不仅是放置节点，而是利用了仿真器底层的 popnet 片上网络模型（NoC）。popnet.log 记录了数据包（Packet）和微片

（Flit）在路由器（Router）之间的每一跳传输，证明系统正确模拟了 XY 路由或类似的死锁避免协议。

通信协议分析：

体现：代码使用了 `InterChiplet::sendMessage` 和 `InterChiplet::receiveMessage` 接口。

仿真器的阻塞式通信机制。CPU 发送数据后，必须等待 GPU 处理并回传，这种“握手”机制确保了异构核心间的数据一致性。



要求2.构建专用 Benchmark 测试芯粒互联性能

构建了cic文件，然后对带宽饱和度和大包传输的稳定性、性能指标、数据完整性验证做了测试。



要求3.提出至少一种改进芯粒互联模块的建议。 实施改进建议并验证性能提升。

提出改进：对同一芯粒的多条小消息进行批量聚合，实现一次 sendMessage 传输 metadata + 多个切片长度 + 大 payload，再一次receiveMessage 回收全部结果，减少控制开销/握手次数，提高有效带宽。

改进情况在cica文件里面。

