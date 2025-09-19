import re
import matplotlib.pyplot as plt


def extract_errors_from_log(log_file):
    IPA_Grad = []
    EGNN_Grad = []

    # 正则表达式模式匹配
    IPA_Grad_pattern = r"Total IPA Grad Norm: (\d+\.\d+)"
    EGNN_Grad_pattern = r"Total EGNN Grad Norm: (\d+\.\d+)"

    with open(log_file, "r", encoding="utf-8") as f:
        log_content = f.read()

        ipa_matches = re.finditer(IPA_Grad_pattern, log_content)
        egnn_matches = re.finditer(EGNN_Grad_pattern, log_content)

        for match in ipa_matches:
            IPA_Grad.append(float(match.group(1)))

        for match in egnn_matches:
            EGNN_Grad.append(float(match.group(1)))

    return IPA_Grad, EGNN_Grad


def plot_error_progression(IPA_Grad, EGNN_Grad):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(IPA_Grad, label="IPA_Grad")
    plt.title("IPA_Grad", fontdict={"size": 22})
    plt.xlabel("loop", fontdict={"size": 22})
    plt.ylabel("Grad", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(EGNN_Grad, label="EGNN_Grad")
    plt.title("EGNN_Grad", fontdict={"size": 22})
    plt.xlabel("loop", fontdict={"size": 22})
    plt.ylabel("Grad", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.savefig("Grad.png", dpi=300)
    plt.close()


# 主程序
if __name__ == "__main__":
    log_file = "377275.out"  # 替换为你的日志文件路径

    # 提取数据
    IPA_Grad, EGNN_Grad = extract_errors_from_log(log_file)

    # 绘制图表
    plot_error_progression(IPA_Grad, EGNN_Grad)
