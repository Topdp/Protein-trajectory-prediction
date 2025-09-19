# eval.py
import torch
from tqdm import tqdm
from main.compute_loss import compute_loss


def eval(model, val_loader, criterion, config, epoch):
    """
    单轮验证
    """
    model.eval()
    val_total_loss = 0.0
    val_atom_loss = 0.0
    val_torsion_loss = 0.0
    val_dist_loss = 0.0
    val_recon_loss = 0.0

    device = config.device

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]"):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(batch, config.pred_steps)
            total_loss, atom_loss, dist_loss, torsion_loss, recon_loss = compute_loss(
                outputs, batch, criterion, epoch, config
            )

            val_total_loss += total_loss.item()
            val_atom_loss += atom_loss.item()
            val_torsion_loss += torsion_loss.item()
            val_dist_loss += dist_loss.item()
            val_recon_loss += recon_loss.item()

    avg_val_total = val_total_loss / len(val_loader)
    avg_val_atom = val_atom_loss / len(val_loader)
    avg_val_torsion = val_torsion_loss / len(val_loader)
    avg_val_dist = val_dist_loss / len(val_loader)
    avg_val_recon = val_recon_loss / len(val_loader) 

    return (
        avg_val_total,
        avg_val_atom,
        avg_val_torsion,
        avg_val_dist,
        avg_val_recon,
    )