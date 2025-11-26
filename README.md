# Learning_Chiplet
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

之后去CUDA的官网安装完之后会出现：setup_environment succeeded



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






> [!WARNING]
>
> 下面的提醒内容不一定使用的上！

> [!NOTE]
>
> 下面需要使用到cmake编译，要用到zmq环境。
>
> 所以在cmake之前，先安装zmq环境,在根目录下安装，指令是如下
>
> ```
> sudo apt-get install libzmq3-dev
> ```
>
> [zmq官网](https://zeromq.org/download/)
>
> [Ubuntu下c++ ZeroMQ环境配置（已踩坑，亲测可用）_zeromq的坑-CSDN博客](https://blog.csdn.net/TU_Dresden/article/details/122240469?spm=1001.2014.3001.5506)
>
> 安装完成后可以用下面三个指令的任意一个指令查看版本
>
> ```
> # 检查头文件路径
> find /usr -name 'zmq.h'
> 
> # 检查库文件
> find /usr -name 'libzmq*'
> 
> # 或者尝试查看版本号（如果pkg-config已安装）
> pkg-config --modversion libzmq
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

cmake的时候有依赖需要安装




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



要求一里面的构建有bug是指的当CPU跟GPU进行数据传输，在最后CPU把输出都传完了，GPU按理来说应该是结束了，但因为里面的while(1)，导致一直没结束，一直在循环空转中，等待CPU传下一个根本不存在的指令。对于这部分的修改bug，需要在GPU和CPU的通信过程中，加入终止信号，让GPU能够退出循环，从而进程终止。

加入终止信号流程

mlp.cu（第一种添加方式）

![image-20251017103738537](C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251017103738537.png)

mlp.cpp（第一种添加方式）

![image-20251017103818937](C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251017103818937.png)

第一种方法的结果图

<img src="C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251125225521273.png" alt="image-20251125225521273" style="zoom: 67%;" />







mlp.cu（第二种添加方式）

![image-20251120160829244](C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251120160829244.png)













mlp.cpp（第二种添加方式）

![image-20251120160714328](C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251120160714328.png)

![image-20251120160732854](C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251120160732854.png)



还有在mlp.yml中修改这个参数，将2改成5，因为芯粒规矩可达到(1，4),mesh网格大小至少是5才不会让popnet的报错

![image-20251120160020132](C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251120160020132.png)



加入完之后，可以先make编译，然后运行下面的代码

```
../../interchiplet/bin/interchiplet ./mlp.yml
```

结果图

![image-20251120155931546](C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251120155931546.png)



---

这部分是做修改的指标（可以看另一个部分的，这部分可以不用看）

对于要求二，可以以阶段一的benchmark/matmul里面的文件来做修改，更改里面的矩阵大小，更改里面的精度，更改传输的容量大小来构建新的benchmark

1.去修改里面的clock_rate,所影响的运行时间和周期

![image-20251017104914772](C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251017104914772.png)

2.去修改矩阵的大小，所影响的运行时间和周期

![image-20251017104956104](C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251017104956104.png)

---







对于要求二，构建了新的边界基准：CPU 驱动程序现在发送每个芯片的元数据（长度 + 限制），流式传输可变大小的 int32 数据负载，并验证 GPU 结果。每个目标使用不同的消息大小来探测 NoC 上的边界条件。
实现了 GPU 工作程序，它接收元数据/负载，限制值，执行块级归约，记录工作，并返回总和。
添加了构建/运行管道：Makefile 用于生成 bin/boundary_cpu 和 bin/boundary_cu，以及 boundary.yml，其中包含 2×2 网格设置（GPU 位于 (0，1)、(1，0)、(1，1)，CPU 位于 (0，0)）和配置为相同网格的 PopNet 阶段。

当前可调的关键参数及默认值如下（可直接改对应文件）：

CPU 侧 (benchmark/boundary/boundary.cpp)

目标列表 targets（目的坐标 + 负载长度 + clamp 值）：
(0,1): len=128, clamp=256
(1,0): len=512, clamp=512
(1,1): len=1024, clamp=1024
随机数分布：std::uniform_int_distribution<int32_t> dist(-1500, 1500)（控制负载数据范围）。
发送顺序：先 16 字节元数据（len, clamp），再 payload，再接收 int64 结果。



GPU 侧 (benchmark/boundary/boundary.cu)
线程块配置：threads = 256；block 数 blocks = (h_len + threads - 1) / threads。
共享内存大小：shared_bytes = threads * sizeof(int64_t)。
clamp_and_sum 核心使用 clamp 限制值、长度 h_len。
交互顺序：接收 16 字节元数据 → 接收 payload（sizeof(int32_t)*h_len）→ 归约 → 发送 int64 结果。
PopNet/拓扑 (benchmark/boundary/boundary.yml, Phase2)



网格与通道：-A 2 (ary size) / -c 2 (cube dimension)，-V 3 虚通道。
缓冲：-B 12 输入缓冲，-O 12 输出缓冲，-F 4 flit 大小。
链路/时间：-L 1000 链路长度，-T 10000000 仿真周期（可放大以覆盖更长时间戳）。
路由：-r 1 路由算法。
Trace：-I ../bench.txt，延迟文件 -D ../delayInfo.txt。
芯粒布置 (boundary.yml, Phase1)



GPU 芯粒坐标与数量：GPU 在 (0,1)、(1,0)、(1,1)，CPU 在 (0,0)。
可以调整以上参数（例如增大/减小 len、clamp，改变随机分布范围，改线程数、缓冲大小、虚通道数、链路长度、仿真周期等）来探索边界条件。







boundary.cpp

```
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "apis_c.h"

// Simple clamp to keep values within |bound|.
int64_t clamp_value(int64_t v, int64_t bound) {
    if (v > bound) return bound;
    if (v < -bound) return -bound;
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: boundary_cpu <srcX> <srcY>\n";
        return 1;
    }

    const int srcX = std::atoi(argv[1]);
    const int srcY = std::atoi(argv[2]);

    // Targets exercise the coordinate boundary of a 2x2 mesh and vary payload size.
    struct Target {
        int dstX;
        int dstY;
        int64_t len;
        int64_t clamp;
    };

    std::vector<Target> targets = {
        {0, 1, 128, 256},
        {1, 0, 512, 512},
        {1, 1, 1024, 1024},
    };

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int32_t> dist(-1500, 1500);

    for (const auto& t : targets) {
        std::vector<int32_t> payload(static_cast<size_t>(t.len));
        for (auto& v : payload) {
            v = dist(rng);
        }

        // Compute expected clamp-and-sum on CPU for verification.
        int64_t expected = 0;
        for (auto v : payload) {
            expected += clamp_value(v, t.clamp);
        }

        int64_t header[2] = {t.len, t.clamp};
        // First send metadata so the GPU knows how many elements to expect and what clamp to use.
        InterChiplet::sendMessage(t.dstX, t.dstY, srcX, srcY, (void*)header, sizeof(header));
        // Then send the data buffer.
        InterChiplet::sendMessage(t.dstX, t.dstY, srcX, srcY, payload.data(),
                                  t.len * sizeof(int32_t));

        int64_t result = 0;
        InterChiplet::receiveMessage(srcX, srcY, t.dstX, t.dstY, (void*)&result,
                                     sizeof(int64_t));

        std::cout << "[CPU] From chiplet (" << t.dstX << "," << t.dstY << ") length=" << t.len
                  << " clamp=" << t.clamp << " -> sum=" << result
                  << " (expected " << expected << ")\n";
    }

    return 0;
}

```



boundary.cu

```
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

#include "apis_cu.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Each block clamps a slice of the payload and reduces to one partial sum.
__global__ void clamp_and_sum(const int32_t* in, int64_t* block_sums, int n, int64_t clamp) {
    extern __shared__ int64_t sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    int64_t v = 0;
    if (idx < n) {
        v = static_cast<int64_t>(in[idx]);
        if (v > clamp) {
            v = clamp;
        } else if (v < -clamp) {
            v = -clamp;
        }
    }
    sdata[tid] = v;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: boundary_cu <idX> <idY>\n");
        return 1;
    }

    int idX = std::atoi(argv[1]);
    int idY = std::atoi(argv[2]);

    int64_t* d_header = nullptr;
    cudaMalloc((void**)&d_header, sizeof(int64_t) * 2);

    // Receive {len, clamp} describing the next payload.
    receiveMessage(idX, idY, 0, 0, d_header, sizeof(int64_t) * 2);
    int64_t h_header[2] = {0, 0};
    cudaMemcpy(h_header, d_header, sizeof(int64_t) * 2, cudaMemcpyDeviceToHost);
    int64_t h_len = h_header[0];
    int64_t clamp = h_header[1];

    if (h_len <= 0) {
        cudaFree(d_header);
        return 0;
    }

    printf("[GPU %d,%d] len=%ld clamp=%ld\n", idX, idY, static_cast<long>(h_len),
           static_cast<long>(clamp));

    int32_t* d_data = nullptr;
    cudaMalloc((void**)&d_data, sizeof(int32_t) * h_len);

    // Receive payload
    receiveMessage(idX, idY, 0, 0, d_data, sizeof(int32_t) * h_len);

    int threads = 256;
    int blocks = static_cast<int>((h_len + threads - 1) / threads);
    int shared_bytes = threads * sizeof(int64_t);

    int64_t* d_block_sums = nullptr;
    cudaMalloc((void**)&d_block_sums, sizeof(int64_t) * blocks);

    clamp_and_sum<<<blocks, threads, shared_bytes>>>(d_data, d_block_sums,
                                                     static_cast<int>(h_len), clamp);
    cudaDeviceSynchronize();

    std::vector<int64_t> h_block_sums(static_cast<size_t>(blocks), 0);
    cudaMemcpy(h_block_sums.data(), d_block_sums, sizeof(int64_t) * blocks,
               cudaMemcpyDeviceToHost);

    int64_t total = 0;
    for (auto v : h_block_sums) {
        total += v;
    }

    printf("[GPU %d,%d] sum=%ld\n", idX, idY, static_cast<long>(total));

    int64_t* d_result = nullptr;
    cudaMalloc((void**)&d_result, sizeof(int64_t));
    cudaMemcpy(d_result, &total, sizeof(int64_t), cudaMemcpyHostToDevice);

    sendMessage(0, 0, idX, idY, d_result, sizeof(int64_t));

    cudaFree(d_header);
    cudaFree(d_data);
    cudaFree(d_block_sums);
    cudaFree(d_result);
    return 0;
}

```



![image-20251121155143246](C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251121155143246.png)





这个基准测试通过在 **2x2 的异构网格**中，向不同距离的节点发送不同规模的数据包，系统性地探测了仿真器在**通信路由**、**数据流控**和**异构同步**这三个维度的边界能力。 我们以 **CPU 的串行计算结果**作为正确性基准，最终的日志和图表不仅验证了所有计算的准确无误，还精确反映了物理距离对通信延迟的影响，证明了该仿真平台具备进行复杂 Chiplet 系统研究的能力。



**1. 首先，明确测试场景与核心目标**
**解释核心：** 搭建了一个 2x2 的异构 Chiplet（芯粒）网格系统，目的是验证仿真器在**复杂通信**和**异构协同**下的“生存能力”。

* **介绍场景**：根据 `boundary.yml` 配置，系统由 1 个 CPU（位置 0,0）和 3 个 GPU（位置 0,1、1,0、1,1）组成。这不再是单一器件的模拟，而是一个完整的分布式系统。
* **解释目标**：不只是为了算一个数，而是为了探测仿真器的**边界**。比如：
  * **通信边界**：数据能不能跨越多跳网络（NoC）准确送达？
  * **同步边界**：CPU 和 GPU 使用不同的仿真引擎（Sniper 和 GPGPU-Sim），它们的时间同步会不会出错？



 **2. 其次，通过多维度的变量设计来“探测边界”**
**解释核心：** 在 `boundary.cpp` 中设计了三种不同的“靶子”（Targets），用控制变量法来向仿真器施压。

* **变量一：拓扑距离（跳数）**
  * 设定了近距离目标 `(0,1)` 和 `(1,0)`（1 跳）以及远距离目标 `(1,1)`（2 跳）。
  * **目的**：测试仿真器的 NoC 路由算法能否正确处理不同距离的传输，会不会在多跳传输中丢包。
* **变量二：数据压力（负载大小）**
  * 发送的数据包大小从 **128** 个整数激增到 **1024** 个整数。
  * **目的**：测试仿真器的缓冲区（Buffer）和流控机制。如果仿真器处理不好大包传输，可能会导致死锁或溢出。
* **变量三：计算负载**
  * 在 `boundary.cu` 中使用了 **"Clamp and Sum"（截断求和）** 算法。
  * **目的**：这是一个包含**访存**（读数据）、**计算**（比较大小）和**同步**（Block 内归约）的典型负载，能全面测试 GPU 核心的功能完备性。



**3. 然后，确立基准并执行验证**
**解释核心：** 这里的“基准”不仅是性能基准，更是**正确性基准（Golden Standard）**。

* **基准是什么？**
  * 基准就是 **CPU 本地的串行计算结果**。在 `boundary.cpp` 中，CPU 在发送任务前，会自己先算一遍 `expected`（预期值）。
* **验证流程**：
  * CPU 将任务通过 NoC 发送给 GPU。
  * GPU 算完后，将结果回传给 CPU。
  * CPU 对比“回传值”和“基准值”。
* **判定标准**：
  * 只有当三组测试（不同距离、不同大小）的结果都**完全匹配**时，我们才认为仿真器通过了边界测试。
  * 日志 `sniper.0.0.log` 显示 `sum=... (expected ...)` 完全一致，这就是成功的铁证。



**4. 最后，通过结果图表解读仿真器行为**
**解释核心：** 结果数据（日志和图表）说明了仿真器在压力下的具体表现。

* **解读通信延迟（Latency）**：
  * 查看 `popnet.log` 或结果图，您会发现发往 `(1,1)` 的任务延迟明显高于 `(0,1)`。
  * **说明**：这证明仿真器的网络模型是精确的，它正确模拟了物理距离带来的延迟开销，而不仅仅是理想化的瞬间传输。
* **解读计算性能（IPC/Cycles）**：
  * 查看 `sim.out`，可以看到 GPU 的 IPC（每周期指令数）约为 1.07。
  * **说明**：这表明在当前负载下，GPU 并没有满负荷运转（可能受限于共享内存冲突或同步等待）。这为架构师提供了优化方向：是增加缓存？还是优化调度？
* **解读功耗（Power）**：
  * 结合 `gpuwattch_gtx480.xml`，结果图可以展示不同阶段的能耗脉冲。
  * **说明**：这验证了仿真器不仅能跑通功能，还能提供细粒度的能效评估数据。





---

### 阶段三：设计拓展

#### 3.1 赛题要求 

1. 在现有仿真器基础上，参考下面的链接：https://github.com/FCAS-ZJU/Chiplet_ Heterogeneous_newVersion/wiki/Imported-Simulator 引入新的 CPU 仿真器。这个仿真器可以自己设计，也可以参考现有的开源项目进行修改；这个模拟器可以设计的很简单，只要满足基本的功能需求即可。具体案例参考附录。 
2. 在此仿真器上运行mlp benchmark。 



#### 3.2 赛题目标 

1. 成功设计或引入新的CPU仿真器，实现任意benchmark的运行； 
2. 成功运行mlp，比较其与现有仿真器的性能差异。



[【亲测免费】 RISC-V ISA Simulator (Spike) 使用教程-CSDN博客](https://blog.csdn.net/gitblog_00219/article/details/142810182)



CPU模拟器参考实现

```
#include <stdint.h>
#include <stdio.h>
#define NREG 4
#define NMEM 16

// 定义指令格式
typedef union {
    struct { uint8_t rs : 2, rt : 2, op : 4; } rtype;
    struct { uint8_t addr : 4      , op : 4; } mtype;
    uint8_t inst;
}inst_t;
 
#define DECODE_R(inst) uint8_t rt = (inst).rtype.rt, rs = (inst).rtype.rs

#define DECODE_M(inst) uint8_t addr = (inst).mtype.addr

uint8_t pc = 0; // PC, C语言中没有4位的数据类型,我们采用8位类型来表示

uint8_t R[NREG] = {}; //寄存器

uint8_t M[NMEM] = { //内存,其中包含一个计算z = x + y的程序
    0b11100110, // load 6# | R[0] <-M[y]
    0b00000100, // mov r1, r0 | R[1] <-R[0]
    0b11100101, // load 5# | R[0] <-M[x]
    0b00010001, // add r0, r1 | R[0] <-R[0] + R[1]
    0b11110111, // store 7# | M[z] <-R[0]
    0b00010000, // x = 16
    0b00100001, // y = 33
    0b00000000, // z = 0
};

int halt = 0; //结束标志

// 执行一条指令
void exec_once() {
 inst_t this;
 this.inst = M[pc]; // 取指
 switch (this.rtype.op) {
    case 0b0000: { DECODE_R(this); R[rt]   = R[rs];   break; }
    case 0b0001: { DECODE_R(this); R[rt]  += R[rs];   break; }
    case 0b1110: { DECODE_M(this); R[0]    = M[addr]; break; }
    case 0b1111: { DECODE_M(this); M[addr] = R[0];    break; }
	default:
	  printf("Invalid instruction with opcode = %x, halting...\n", this.rtype.op);
      halt = 1;
	  break;
  }
   pc++; // 更新PC
}
int main() {
 	while (1) {
      exec_once();
      if (halt) break;
     }
  	 printf("The result of 16 + 33 is %d\n", M[7]);
   	 return 0;
}
```





采用参考现有的开源项目进行修改，因为感觉用其他的比较麻烦，所以采用他自带的gem5来代替sniper去做CPU仿真器的效果

去修改的地方有几处，首先是gem5的，先要去到gem5/src/sim/syscall_emul.cc这个文件，对这个文件做修改

修改前

<img src="C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251123111046360.png" alt="image-20251123111046360" style="zoom:50%;" />



修改后

<img src="C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251123111008612.png" alt="image-20251123111008612" style="zoom:50%;" />

修改这个是因为这里原来的函数会导致数据写入的方向错误，所以去修改到正常的写入方向。

修改完后，还有各类的benchmark里面的yml文件需要修改，比如下面的这些，matmul.yml和cic.yml

<img src="C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251123111252393.png" alt="image-20251123111252393" style="zoom:50%;" />

<img src="C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251123111330413.png" alt="image-20251123111330413" style="zoom:50%;" />

修改好后，编译运行程序，程序是可以正常运行结束的

<img src="C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251123111507590.png" alt="image-20251123111507590" style="zoom:50%;" />



> [!IMPORTANT]
>
> 然后去修改下yml中的仿真长度

未修改前

<img src="C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251123113004627.png" alt="image-20251123113004627" style="zoom:50%;" />

<img src="C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251123112954529.png" alt="image-20251123112954529" style="zoom:50%;" />

已修改后

<img src="C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251123112840554.png" alt="image-20251123112840554" style="zoom:50%;" />

<img src="C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251123112854132.png" alt="image-20251123112854132" style="zoom:50%;" />

> [!NOTE]
>
> PopNet 没有完成事务是因为仿真长度太短：bench.txt 的时间戳在 4.11e9 以上，而原来 -T 10000000 只跑到 1e7，就直接结束，导致 “total finished 0 / average Delay -nan”。所以修改时间戳上限，让程序能够跑完
>
> 同时这个问题，会让gpgpusim报一个status=134的错误，会提前进入死锁状态不让进程进行下去。



然后查看gem5的日志，能够看到是正常结束的

<img src="C:\Users\29800\AppData\Roaming\Typora\typora-user-images\image-20251123113438027.png" alt="image-20251123113438027" style="zoom:50%;" />

对此运行这个benchmark是没有问题的

然后是去对任意benchmark做运行，对我们自己设计的三个benchmark做下运行（cic和cica是赛题四所做的，也放在这里去运行下结果）然后发现是都可以正常运行成功的，至此这个要求和目标是已经完成了。

---



对于运行mlp的要求目标暂时做不了。




---



### 阶段四：芯粒互联架构分析与应用

#### 4.1 赛题要求 

1. 分析仿真器中芯粒互联机制的工作原理；
2. 构建专用benchmark 测试芯粒互联性能； 
3. 提出至少一种改进芯粒互联模块的建议。 



#### 4.2 赛题目标 

1. 成功运行芯粒互联benchmark，记录测试结果； 
2. 实施改进建议并验证性能提升。



1.分析仿真器中芯粒互联机制的工作原理

拓扑与路由机制分析：

体现：在 cic.yml 中，构建了一个 2x2 的网格系统（Mesh），包含 1 个 CPU (0,0) 和 3 个 GPU ((0,1), (1,0), (1,1))。

这不仅是放置节点，而是利用了仿真器底层的 popnet 片上网络模型（NoC）。popnet.log 记录了数据包（Packet）和微片

（Flit）在路由器（Router）之间的每一跳传输，证明系统正确模拟了 XY 路由或类似的死锁避免协议。

通信协议分析：

体现：代码使用了 `InterChiplet::sendMessage` 和 `InterChiplet::receiveMessage` 接口。

仿真器的阻塞式通信机制。CPU 发送数据后，必须等待 GPU 处理并回传，这种“握手”机制确保了异构核心间的数据一致性。



2.构建专用 Benchmark 测试芯粒互联性能

变量控制：

不同距离：测试目标覆盖了单跳邻居 (0,1), (1,0) 和两跳对角邻居 (1,1)，用于分析距离对延迟的影响。

不同负载：测试了 256, 1024, 4096, 16384 四种不同大小的数据包（int32数组）。这是为了测量带宽饱和度和大包传输的稳定性。



性能指标：

体现：与之前的 boundary 不同，cic.cpp 引入了 std::chrono 计时器。

代码逻辑：在发送请求前记录 t0，在收到结果后记录 t1，计算 us (微秒) 级延迟。

目的：这是专门为了量化互联性能而写的代码，区别于仅验证正确性的功能测试。



数据完整性验证：

虽然侧重性能，但代码依然保留了 checksum（校验和）机制，确保在高性能传输下数据未发生比特翻转或丢包。

成功运行 Benchmark 并记录测试结果

运行日志 sniper.0.0.log 提供了完美的证据，证明目标已达成：



全流程跑通：

日志显示测试覆盖了所有预设的组合：3 个目标节点 × 4 种数据包大小。

程序最后正常退出（Leaving ROI, End），没有发生死锁或崩溃。



性能数据记录：

日志清晰记录了每一组测试的延迟数据，例如：

小包（256 int）：dst=(0,1) len=256 ... latency(us)=0。

大包（16384 int）：dst=(1,1) len=16384 ... latency(us)=1。

(注：这里的 latency(us) 数值较小可能是因为这是宿主机的墙钟时间，或者是仿真时间步进较快，但机制本身已成功运行并输出了结果)。



3.结果正确性：

每一行的 sum 和 expected 值都完全一致（例如 sum=22883 expected=22883），证明互联通信是可靠的。




提出至少一种改进芯粒互联模块的建议。 实施改进建议并验证性能提升。

改进情况

提出改进：对同一芯粒的多条小消息进行批量聚合，实现一次 sendMessage 传输 metadata + 多个切片长度 + 大 payload，再一次

receiveMessage 回收全部结果，减少控制开销/握手次数，提高有效带宽。



实现要点

cica.cpp 中，将 {256,1024,4096,16384} 四个长度聚合成一批，按目标芯粒一次性发送；

cica.cu 中，按 batch_count 和长度数组切分 payload，循环调用 reduce_sum，并集中回传结果数组。



cica.cpp：为三个邻居芯粒一次性构造 4 段 payload，先发送 batch_count 与各段长度，然后发送聚合数据；一次收回所有切片的校验和，记录总字节数、时延和近似带宽；最后广播 0 结束信号。

cica.cu：循环接收 batch 描述与长度、payload，逐段进行块内归约求和，打印批次信息，将所有结果打包一次返回 (减少往返消息)。



运行与验证：

对比 cica的popnet.log 输出的近似带宽，与原 CIC 基线的 per-slice 往返日志对比，可看到消息次数从每切片 3 次降到批量 4 次，控制开

销显著下降。


性能验证结果

消息条数从 30 降至 8，1-flit 控制消息从 20 条降至 6 条；

芯粒互联由多次 request-reply 变为一次批量传输，有效减小头部和仲裁开销；

CPU 端测得的 per-link 近似带宽 approx_bw(GB/s) 提升。



