import re
import matplotlib.pyplot as plt

def extract_errors_from_log(log_file):
    # 初始化存储列表
    train_total_losses = []
    train_atom_losses = []
    train_tor_losses = []
    train_dist_losses = []
    train_rec_losses = []

    val_total_losses = []
    val_atom_losses = []
    val_tor_losses = []
    val_dist_losses = []
    val_rec_losses = []

    current_section = None

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # 检查是否是训练或验证开始
            if line == "训练损失:":
                current_section = "train"
                continue
            elif line == "验证损失:":
                current_section = "val"
                continue

            # 检查是否是新的epoch开始
            if line.startswith("=== Epoch"):
                current_section = None
                continue

            if not current_section:
                continue

            # 尝试匹配各种损失模式
            if current_section == "train":
                if match := re.search(r"总损失:\s*(\d+\.\d+)", line):
                    train_total_losses.append(float(match.group(1)))
                elif match := re.search(r"原子坐标损失:\s*(\d+\.\d+)", line):
                    train_atom_losses.append(float(match.group(1)))
                elif match := re.search(r"扭转角损失:\s*(\d+\.\d+)", line):
                    train_tor_losses.append(float(match.group(1)))
                elif match := re.search(r"距离图损失:\s*(\d+\.\d+)", line):
                    train_dist_losses.append(float(match.group(1)))
                elif match := re.search(r"结构重建损失:\s*(\d+\.\d+)", line):
                    train_rec_losses.append(float(match.group(1)))
            else:  # current_section == "val"
                if match := re.search(r"总损失:\s*(\d+\.\d+)", line):
                    val_total_losses.append(float(match.group(1)))
                elif match := re.search(r"原子坐标损失:\s*(\d+\.\d+)", line):
                    val_atom_losses.append(float(match.group(1)))
                elif match := re.search(r"扭转角损失:\s*(\d+\.\d+)", line):
                    val_tor_losses.append(float(match.group(1)))
                elif match := re.search(r"距离图损失:\s*(\d+\.\d+)", line):
                    val_dist_losses.append(float(match.group(1)))
                elif match := re.search(r"结构重建损失:\s*(\d+\.\d+)", line):
                    val_rec_losses.append(float(match.group(1)))

    # 调试信息
    print(f"训练总损失匹配数量: {len(train_total_losses)}")
    print(f"训练原子坐标损失匹配数量: {len(train_atom_losses)}")
    print(f"训练扭转角损失匹配数量: {len(train_tor_losses)}")
    print(f"训练距离极损失匹配数量: {len(train_dist_losses)}")
    print(f"训练结构重建损失匹配数量: {len(train_rec_losses)}")

    print(f"验证总损失匹配数量: {len(val_total_losses)}")
    print(f"验证原子坐标损失匹配数量: {len(val_atom_losses)}")
    print(f"验证扭转角损失匹配数量: {len(val_tor_losses)}")
    print(f"验证距离图损失匹配数量: {len(val_dist_losses)}")
    print(f"验证结构重建损失匹配数量: {len(val_rec_losses)}")

    return {
        "train": {
            "total": train_total_losses,
            "atom": train_atom_losses,
            "tor": train_tor_losses,
            "dist": train_dist_losses,
            "rec": train_rec_losses,
        },
        "val": {
            "total": val_total_losses,
            "atom": val_atom_losses,
            "tor": val_tor_losses,
            "dist": val_dist_losses,
            "rec": val_rec_losses,
        },
    }


def plot_error_progression(data, log_file):
    train_data = data["train"]
    val_data = data["val"]
    
    # 检查数据是否为空
    if not train_data["total"]:
        print("错误: 没有提取到任何训练损失数据")
        return
        
    if not val_data["total"]:
        print("错误: 没有提取到任何验证损失数据")
        return

    epochs = range(1, len(train_data["total"]) + 1)

    # 创建2x2的子图布局
    plt.figure(figsize=(12, 10))

    # 总损失图
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_data["total"], "b-", label="train")
    plt.plot(epochs, val_data["total"], "r-", label="val")
    plt.title("total loss", fontdict={"size": 16})
    plt.xlabel("Epoch", fontdict={"size": 12})
    plt.ylabel("loss", fontdict={"size": 12})
    plt.legend()
    plt.grid(True)

    # 原子坐标损失图
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_data["atom"], "b-", label="train")
    plt.plot(epochs, val_data["atom"], "r-", label="val")
    plt.title("rmsd loss", fontdict={"size": 16})
    plt.xlabel("Epoch", fontdict={"size": 12})
    plt.ylabel("loss", fontdict={"size": 12})
    plt.legend()
    plt.grid(True)

    # 扭转角损失图
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_data["tor"], "b-", label="train")
    plt.plot(epochs, val_data["tor"], "r-", label="val")
    plt.title("torsion loss", fontdict={"size": 16})
    plt.xlabel("Epoch", fontdict={"size": 12})
    plt.ylabel("loss", fontdict={"size": 12})
    plt.legend()
    plt.grid(True)

    # 距离图损失图
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_data["dist"], "b-", label="train")
    plt.plot(epochs, val_data["dist"], "r-", label="val")
    plt.title("distance loss", fontdict={"size": 16})
    plt.xlabel("Epoch", fontdict={"size": 12})
    plt.ylabel("loss", fontdict={"size": 12})
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"loss_analysis_{log_file}.png", dpi=300)
    plt.close()

    # 单独绘制结构重建损失图
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_data["rec"], "b-", label="train")
    plt.plot(epochs, val_data["rec"], "r-", label="val")
    plt.title("recon loss", fontdict={"size": 16})
    plt.xlabel("Epoch", fontdict={"size": 12})
    plt.ylabel("loss", fontdict={"size": 12})
    plt.legend()
    plt.grid(True)
    plt.savefig(f"loss_rec_{log_file}.png", dpi=300)
    plt.close()


# 主程序
if __name__ == "__main__":
    log_file = "377580.out"  # 替换为你的日志文件路径

    # 提取数据
    loss_data = extract_errors_from_log(log_file)

    # 绘制图表
    plot_error_progression(loss_data, log_file)