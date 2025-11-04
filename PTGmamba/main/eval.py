# eval.py
import torch
from tqdm import tqdm
from compute_loss import compute_loss
from compute_metrics import compute_all_metrics, compute_tm_score, compute_contact_accuracy


def eval(model, val_loader, criterion, config, epoch):
    """
    验证函数
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        config: 配置
        epoch: 当前epoch
    """
    model.eval()
    val_total_loss = 0.0
    val_atom_loss = 0.0
    val_torsion_loss = 0.0
    val_dist_loss = 0.0
    val_recon_loss = 0.0
    val_recon_constraint = 0.0
    val_bb_rmsd = 0.0  # 主链RMSD
    val_sc_rmsd = 0.0  # 侧链RMSD
    
    # 准确率指标累积
    all_rmsd_values = []
    accuracies_sum = {f"acc@{t}A": 0.0 for t in [2.0, 5.0, 10.0]}
    tm_scores_sum = 0.0
    contact_acc_sum = 0.0
    n_batches = 0

    device = config.device

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"验证 Epoch {epoch+1}"):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(batch, config.pred_steps)
            
            total_loss, atom_loss, dist_loss, torsion_loss, recon_loss, recon_constraint, \
                pred_bb_rmsd, pred_sc_rmsd = compute_loss(outputs, batch, criterion, epoch, config)

            val_total_loss += total_loss.item()
            val_atom_loss += atom_loss.item()
            val_torsion_loss += torsion_loss.item()
            val_dist_loss += dist_loss.item()
            val_recon_loss += recon_loss.item()
            val_recon_constraint += recon_constraint.item()
            val_bb_rmsd += pred_bb_rmsd.item()
            val_sc_rmsd += pred_sc_rmsd.item()
            
            # 计算准确率指标
            metrics, rmsd = compute_all_metrics(outputs, batch, thresholds=[2.0, 5.0, 10.0])
            
            all_rmsd_values.append(rmsd.cpu())
            
            for key in accuracies_sum.keys():
                accuracies_sum[key] += metrics[key]
            
            # 计算TM-score
            tm_score = compute_tm_score(rmsd, d0=5.0)
            tm_scores_sum += tm_score
            
            # 计算接触图准确率
            pred_coords = outputs["pred_coords"]
            atom_target = batch["atom_positions_target"]
            pred_atom_mask = batch["atom_mask_target"]
            
            B, T_pred, _, _, _ = atom_target.shape
            atom_target_flat = atom_target.reshape(-1, 3)
            pred_atom_mask_flat = pred_atom_mask.reshape(-1).bool()
            masked_target_flat = atom_target_flat[pred_atom_mask_flat]
            N_valid = pred_coords.shape[2]
            masked_target = masked_target_flat.view(B, T_pred, N_valid, 3)
            
            contact_acc = compute_contact_accuracy(pred_coords, masked_target, threshold=8.0)
            contact_acc_sum += contact_acc
            
            n_batches += 1

    # 计算平均值
    avg_val_total = val_total_loss / len(val_loader)
    avg_val_atom = val_atom_loss / len(val_loader)
    avg_val_torsion = val_torsion_loss / len(val_loader)
    avg_val_dist = val_dist_loss / len(val_loader)
    avg_val_recon = val_recon_loss / len(val_loader)
    avg_val_recon_constraint = val_recon_constraint / len(val_loader)
    avg_val_bb_rmsd = val_bb_rmsd / len(val_loader)
    avg_val_sc_rmsd = val_sc_rmsd / len(val_loader)
    
    avg_accuracies = {key: val / n_batches for key, val in accuracies_sum.items()}
    avg_tm_score = tm_scores_sum / n_batches
    avg_contact_acc = contact_acc_sum / n_batches
    
    # 合并所有RMSD值并计算统计信息
    all_rmsd = torch.cat(all_rmsd_values, dim=0)  # [Total_samples, T_pred]
    rmsd_mean = all_rmsd.mean().item()
    rmsd_std = all_rmsd.std().item()
    rmsd_median = all_rmsd.median().item()

    return {
        # 损失指标
        "total_loss": avg_val_total,
        "atom_loss": avg_val_atom,
        "torsion_loss": avg_val_torsion,
        "dist_loss": avg_val_dist,
        "recon_loss": avg_val_recon,
        "recon_constraint": avg_val_recon_constraint,
        "bb_rmsd": avg_val_bb_rmsd,  # 主链RMSD
        "sc_rmsd": avg_val_sc_rmsd,  # 侧链RMSD
        "rmsd_mean": rmsd_mean,
        "rmsd_std": rmsd_std,
        "rmsd_median": rmsd_median,
        **avg_accuracies,
        "tm_score": avg_tm_score,
        "contact_acc": avg_contact_acc,
    }