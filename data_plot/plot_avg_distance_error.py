import re
import matplotlib.pyplot as plt


def extract_errors_from_log(log_file):
    atom_errors = []
    pool_errors = []
    epochs = []

    # 正则表达式模式匹配
    epoch_pattern = r"Epoch (\d+)/150"
    atom_error_pattern = r"原子级结构重建:[\s\S]*?avg_distance_error: (\d+\.\d+)"
    pool_error_pattern = r"池化级结构重建:[\s\S]*?avg_distance_error: (\d+\.\d+)"

    with open(log_file, "r", encoding="utf-8") as f:
        log_content = f.read()

        epoch_matches = re.finditer(epoch_pattern, log_content)
        atom_matches = re.finditer(atom_error_pattern, log_content)
        pool_matches = re.finditer(pool_error_pattern, log_content)

        # 提取epoch
        for match in epoch_matches:
            epochs.append(int(match.group(1)))

        # 提取原子级误差
        # for match in atom_matches:
        #     atom_errors.append(float(match.group(1)))

        # 提取池化级误差
        for match in pool_matches:
            pool_errors.append(float(match.group(1)))

    return epochs, atom_errors, pool_errors


def plot_error_progression(epochs, atom_errors, pool_errors):
    plt.figure(figsize=(12, 8))

    # 绘制原子级重建误差
    # plt.plot(
    #     epochs,
    #     atom_errors,
    #     label="no pool error",
    #     color="blue",
    #     linewidth=2,
    #     markersize=8,
    # )

    # 绘制池化级重建误差
    plt.plot(
        epochs,
        pool_errors,
        label="pool error",
        color="red",
        linewidth=2,
        markersize=8,
    )

    # 添加标题和标签
    plt.title(
        "structure reconstruct error - train epochs",
        fontsize=16,
    )
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("RMSD (Å)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # 设置坐标轴范围
    plt.xlim(min(epochs) - 0.5, max(epochs) + 0.5)
    # plt.ylim(0, max(max(atom_errors), max(pool_errors)) * 1.1)
    # plt.ylim(0, max(pool_errors)) * 1.1)

    plt.tight_layout()
    plt.savefig("structure_reconstruction_error_progression.png", dpi=300)
    plt.show()


# 主程序
if __name__ == "__main__":
    log_file = "328253.out"  # 替换为你的日志文件路径

    # 提取数据
    epochs, atom_errors, pool_errors = extract_errors_from_log(log_file)

    # 打印提取的数据
    print("提取的数据:")
    print("Epochs:", epochs)
    print("原子级重建误差:", atom_errors)
    print("池化级重建误差:", pool_errors)

    # 绘制图表
    plot_error_progression(epochs, atom_errors, pool_errors)
