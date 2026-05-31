# 计算机视觉大作业：CNN 学习与实践

本项目完成两个实验：

1. MNIST 手写数字识别：手动搭建 LeNet-5，不加载任何预训练模型。
2. CIFAR-10 物体分类：手动搭建 CNN，不加载任何预训练模型。

> 说明：代码默认会从官方地址下载 MNIST 与 CIFAR-10 数据集。如果学校机房或本地网络无法访问，请先手动下载数据集，再通过 `--data-dir` 指向数据目录。

## 1. 环境配置

建议使用 Python 3.9-3.11。安装依赖：

```bash
pip install -r requirements.txt
```

推荐有 NVIDIA GPU 的环境；没有 GPU 也可以运行，但 CIFAR-10 训练会比较慢。

## 2. 运行 MNIST / LeNet-5 实验

```bash
python src/train_mnist.py --epochs 10 --batch-size 128 --lr 0.001 --data-dir .data --output-dir outputs/mnist
```

运行结束后会生成：

- `outputs/mnist/best_model.pth`：测试集准确率最高的模型权重
- `outputs/mnist/history.csv`：每轮训练和测试指标
- `outputs/mnist/loss_curve.png`：损失曲线
- `outputs/mnist/accuracy_curve.png`：准确率曲线
- `outputs/mnist/final_metrics.json`：最终测试结果

## 3. 运行 CIFAR-10 / CNN 实验

```bash
python src/train_cifar10.py --epochs 30 --batch-size 128 --lr 0.001 --data-dir .data --output-dir outputs/cifar10
```

运行结束后会生成：

- `outputs/cifar10/best_model.pth`
- `outputs/cifar10/history.csv`
- `outputs/cifar10/loss_curve.png`
- `outputs/cifar10/accuracy_curve.png`
- `outputs/cifar10/final_metrics.json`

## 4. 一键运行两个实验

```bash
python src/run_all.py --data-dir .data --output-dir outputs
```

默认参数：MNIST 训练 10 轮，CIFAR-10 训练 30 轮。可在命令行中修改轮数、学习率和批大小。

## 5. 复现实验报告中的曲线和指标

报告中给出的是默认配置下的参考结果。正式提交前建议在你的电脑上运行训练脚本，并把 `outputs/mnist` 和 `outputs/cifar10` 中生成的真实曲线与准确率替换到报告中。

## 6. 项目结构

```text
cv_cnn_project/
├── README.md
├── requirements.txt
├── src/
│   ├── datasets.py       # 不依赖 torchvision 的 MNIST / CIFAR-10 数据加载
│   ├── models.py         # LeNet-5 与 CIFAR-10 CNN 网络结构
│   ├── train_utils.py    # 训练、测试、保存曲线等公共函数
│   ├── train_mnist.py    # MNIST 训练入口
│   ├── train_cifar10.py  # CIFAR-10 训练入口
│   └── run_all.py        # 一键运行两个实验
├── report/
│   ├── CV_CNN_Report.pdf # 实验报告，可按“学号_姓名_CV_CNN.pdf”重命名
│   └── CV_CNN_Report.docx
└── outputs/
    ├── mnist/
    └── cifar10/
```

## 7. 注意事项

- 本项目只使用 PyTorch 基础模块搭建网络，不加载 torchvision 内置模型或其他预训练 CNN。
- CIFAR-10 实验使用随机裁剪、随机水平翻转和 BatchNorm 提升泛化能力。
- 若训练准确率远高于测试准确率，说明可能过拟合，可增大 Dropout、使用数据增强或减少训练轮数。
