import torch
import Config as cfg
import time

from v1_loss_comp import ProteinLoss

config = cfg.Config()


def eval(model, val_loader, epoch):
    model.eval()
    val_loss = 0
    all_recon_loss = 0
    all_coords_loss = 0
    start_time = time.time()

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(config.device)

            # 模型推理
            pred_dict = model.generate(batch, pred_steps=config.pred_steps)

            loss = ProteinLoss()
            total_loss, recon_loss, coords_loss = loss(pred_dict, batch, config, epoch)

            val_loss += total_loss.item()
            all_recon_loss += recon_loss.item()
            all_coords_loss += coords_loss.item()

    avg_val_loss = val_loss / len(val_loader)
    all_recon_loss = all_recon_loss / len(val_loader)
    all_coords_loss = all_coords_loss / len(val_loader)
    print(f"验证耗时: {time.time()-start_time:.2f}秒")
    print(f"平均验证损失: {avg_val_loss:.6f}")
    return avg_val_loss, all_recon_loss, all_coords_loss
