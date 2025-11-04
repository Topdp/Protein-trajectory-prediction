import re
import matplotlib.pyplot as plt

def extract_errors_from_log(log_file):
    train_total_losses = []
    train_atom_losses = []
    train_tor_losses = []
    train_dist_losses = []
    train_rec_losses = []
    train_physical_loss = []

    val_total_losses = []
    val_atom_losses = []
    val_tor_losses = []
    val_dist_losses = []
    val_rec_losses = []
    val_physical_loss = []

    current_section = None

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # 检查是否是训练或验证开始
            if line == "[Training Metrics]":
                current_section = "train"
                continue
            elif line == "[Validation Metrics]":
                current_section = "val"
                continue

            # 检查是否是RMSD统计开始（表示验证损失结束）
            if line.startswith("[RMSD Statistics"):
                current_section = None
                continue
            
            # 检查是否是新的epoch开始
            if line.startswith("=== Epoch") or line.startswith("================"):
                if "Epoch" in line and "Validation" not in line:
                    current_section = None
                continue

            if not current_section:
                continue

            # 匹配各种损失
            if current_section == "train":
                if match := re.search(r"Total Loss:\s+(\d+\.\d+)", line):
                    train_total_losses.append(float(match.group(1)))
                elif match := re.search(r"(?:Atom Loss|RMSD Loss):\s+(\d+\.\d+)", line):
                    train_atom_losses.append(float(match.group(1)))
                elif match := re.search(r"Torsion Loss:\s+(\d+\.\d+)", line):
                    train_tor_losses.append(float(match.group(1)))
                elif match := re.search(r"Distance Loss:\s+(\d+\.\d+)", line):
                    train_dist_losses.append(float(match.group(1)))
                elif match := re.search(r"Reconstruction:\s+(\d+\.\d+)", line):
                    train_rec_losses.append(float(match.group(1)))
                elif match := re.search(r"(?:Physical Violation|Clash Loss):\s+(\d+\.\d+)", line):
                    train_physical_loss.append(float(match.group(1)))
            else:  # current_section == "val"
                if match := re.search(r"Total Loss:\s+(\d+\.\d+)", line):
                    val_total_losses.append(float(match.group(1)))
                elif match := re.search(r"(?:Atom Loss|RMSD Loss):\s+(\d+\.\d+)", line):
                    val_atom_losses.append(float(match.group(1)))
                elif match := re.search(r"Torsion Loss:\s+(\d+\.\d+)", line):
                    val_tor_losses.append(float(match.group(1)))
                elif match := re.search(r"Distance Loss:\s+(\d+\.\d+)", line):
                    val_dist_losses.append(float(match.group(1)))
                elif match := re.search(r"Reconstruction:\s+(\d+\.\d+)", line):
                    val_rec_losses.append(float(match.group(1)))
                elif match := re.search(r"(?:Physical Violation|Clash Loss):\s+(\d+\.\d+)", line):
                    val_physical_loss.append(float(match.group(1)))

    # 提取统计
    print(f"Train - Total: {len(train_total_losses)}, RMSD: {len(train_atom_losses)}, "
          f"Torsion: {len(train_tor_losses)}, Distance: {len(train_dist_losses)}, "
          f"Recon: {len(train_rec_losses)}, Physical: {len(train_physical_loss)}")
    print(f"Val   - Total: {len(val_total_losses)}, RMSD: {len(val_atom_losses)}, "
          f"Torsion: {len(val_tor_losses)}, Distance: {len(val_dist_losses)}, "
          f"Recon: {len(val_rec_losses)}, Physical: {len(val_physical_loss)}")
    
    # 检查是否成功提取数据
    if len(train_atom_losses) == 0 and len(train_total_losses) > 0:
        print("\n⚠️ 警告：RMSD损失未提取到，但总损失已提取")
        print("   检查日志格式是否为'RMSD Loss'或'Atom Loss'")
    
    if len(val_atom_losses) == 0 and len(val_total_losses) > 0:
        print("\n⚠️ 警告：验证RMSD损失未提取到")
        print("   将尝试继续绘图，但RMSD子图可能为空")

    return {
        "train": {
            "total": train_total_losses,
            "atom": train_atom_losses,
            "tor": train_tor_losses,
            "dist": train_dist_losses,
            "rec": train_rec_losses,
            "physical": train_physical_loss,
        },
        "val": {
            "total": val_total_losses,
            "atom": val_atom_losses,
            "tor": val_tor_losses,
            "dist": val_dist_losses,
            "rec": val_rec_losses,
            "physical": val_physical_loss,
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

    # 创建2x3的子图布局（5个损失图+1个空白）
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training and Validation Loss - {log_file}', fontsize=16, fontweight='bold')

    # 总损失图
    ax = axes[0, 0]
    ax.plot(epochs, train_data["total"], "b-", label="Train")
    ax.plot(epochs, val_data["total"], "r-", label="Val")
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSD损失图
    ax = axes[0, 1]
    train_atom = train_data["atom"] if len(train_data["atom"]) > 0 else [0] * len(epochs)
    val_atom = val_data["atom"] if len(val_data["atom"]) > 0 else [0] * len(epochs)
    ax.plot(epochs, train_atom, "b-", label="Train")
    ax.plot(epochs, val_atom, "r-", label="Val")
    ax.set_title("RMSD Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Torsion Loss
    ax = axes[0, 2]
    train_tor = train_data["tor"] if len(train_data["tor"]) > 0 else [0] * len(epochs)
    val_tor = val_data["tor"] if len(val_data["tor"]) > 0 else [0] * len(epochs)
    ax.plot(epochs, train_tor, "b-", label="Train")
    ax.plot(epochs, val_tor, "r-", label="Val")
    ax.set_title("Torsion Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Distance Loss
    ax = axes[1, 0]
    train_dist = train_data["dist"] if len(train_data["dist"]) > 0 else [0] * len(epochs)
    val_dist = val_data["dist"] if len(val_data["dist"]) > 0 else [0] * len(epochs)
    ax.plot(epochs, train_dist, "b-", label="Train")
    ax.plot(epochs, val_dist, "r-", label="Val")
    ax.set_title("Distance Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reconstruction Loss
    ax = axes[1, 1]
    train_rec = train_data["rec"] if len(train_data["rec"]) > 0 else [0] * len(epochs)
    val_rec = val_data["rec"] if len(val_data["rec"]) > 0 else [0] * len(epochs)
    ax.plot(epochs, train_rec, "b-", label="Train")
    ax.plot(epochs, val_rec, "r-", label="Val")
    ax.set_title("Reconstruction Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Physical Violation
    ax = axes[1, 2]
    train_phys = train_data["physical"] if len(train_data["physical"]) > 0 else [0] * len(epochs)
    val_phys = val_data["physical"] if len(val_data["physical"]) > 0 else [0] * len(epochs)
    ax.plot(epochs, train_phys, "b-", label="Train")
    ax.plot(epochs, val_phys, "r-", label="Val")
    ax.set_title("Physical Violation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Violation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"loss_analysis_{log_file}.png", dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: loss_analysis_{log_file}.png")
    plt.close()


# Main
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "423158.out"
    
    print(f"Analyzing: {log_file}")
    print("="*60)

    loss_data = extract_errors_from_log(log_file)
    
    if not loss_data["train"]["total"]:
        print("Error: No loss data extracted")
        sys.exit(1)
    
    print(f"\n✓ Extracted {len(loss_data['train']['total'])} epochs")
    print("="*60)

    plot_error_progression(loss_data, log_file)
    
    print("\n✓ Done!")
    print("Layout: 2x3 (6 subplots)")
    print("  - Total Loss")
    print("  - RMSD Loss")
    print("  - Torsion Loss")
    print("  - Distance Loss")
    print("  - Reconstruction Loss")
    print("  - Physical Violation")