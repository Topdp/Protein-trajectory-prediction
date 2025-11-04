"""
提供全局缓存的VdW半径和键参数查找表，避免重复构建。
"""

import torch
from util.chemical import VDW_RADIUS, get_atom14_element, get_atom14_bonds


# 全局缓存变量
_vdw_table_cache = {}  # {device: tensor}
_bond_table_cache = {}  # {device: (bond_table, bond_mask)}


def get_vdw_table(device):
    """
    获取VdW半径查找表, 带缓存
    
    Args:
        device: torch.device 或 'cuda'/'cpu'
        
    Returns:
        vdw_table: [20, 14] 张量
        
    Example:
        vdw_table = get_vdw_table(device)
        radii = vdw_table[aatype, atom_idx]
    """
    device_key = str(device)
    
    if device_key not in _vdw_table_cache:
        # 构建VdW半径查找表
        vdw_table = torch.zeros(20, 14, device=device)
        for aa_idx in range(20):
            for atom_idx in range(14):
                element = get_atom14_element(aa_idx, atom_idx)
                if element is not None:
                    vdw_table[aa_idx, atom_idx] = VDW_RADIUS.get(element, 1.70)
        
        _vdw_table_cache[device_key] = vdw_table
    
    return _vdw_table_cache[device_key]


def get_bond_tables(device):
    """
    获取键参数查找表,缓存
    
    Args:
        
    Returns:
        bond_table: [20, max_bonds, 4] 张量 (atom1, atom2, length, stddev)
        bond_mask: [20, max_bonds] 布尔张量（键有效性）
        
    Example:
        bond_table, bond_mask = get_bond_tables(device)
        bonds_info = bond_table[aatype]  # [batch, max_bonds, 4]
    """
    device_key = str(device)
    
    if device_key not in _bond_table_cache:
        # 构建键参数查找表
        max_bonds = 15
        bond_table = torch.full(
            (20, max_bonds, 4), -1.0, dtype=torch.float32, device=device
        )
        bond_mask = torch.zeros((20, max_bonds), dtype=torch.bool, device=device)
        
        for aa_idx in range(20):
            bonds = get_atom14_bonds(aa_idx)
            for bond_idx, (atom1, atom2, length, stddev) in enumerate(bonds):
                bond_table[aa_idx, bond_idx] = torch.tensor(
                    [atom1, atom2, length, stddev], device=device
                )
                bond_mask[aa_idx, bond_idx] = True
        
        _bond_table_cache[device_key] = (bond_table, bond_mask)
    
    return _bond_table_cache[device_key]


def clear_cache():
    global _vdw_table_cache, _bond_table_cache
    _vdw_table_cache.clear()
    _bond_table_cache.clear()

