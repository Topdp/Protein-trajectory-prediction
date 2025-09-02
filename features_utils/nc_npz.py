import sys
import mdtraj, os, tempfile, tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import numpy as np
from openfold.np import protein
from openfold.data.data_pipeline import make_protein_features
import Config as cfg

# 数据显示
np.set_printoptions(threshold=np.inf)

config = cfg.Config()


def preprocess(top_name, pdb_name):
    """
    轨迹预处理
    Args
        top_name: the top files name
        pdb_name: find path with pdb name
    """
    traj = mdtraj.load(
        f"./trajectory/{pdb_name}/{config.traj_name}.dcd",
        top=f"./trajectory/{pdb_name}/{pdb_name}.prmtop",
    )
    print(f"Total frames:{traj.n_frames}")
    print(f"Number of residues before processing:{traj.n_residues}")
    print(f"Number of atoms in the system before processing:{traj.n_atoms}")

    solvent_names = []
    # TIP3P水分子（WAT、HOH） 常见离子（Na+、Cl-、K+等）,为空则去除所有溶剂，离子
    print("Removal of water and ions ...")
    traj_protein = traj.remove_solvent(
        exclude=solvent_names,
        inplace=False,
    )

    print(f"Number of residues after processing:{traj_protein.n_residues}")
    print(f"Number of atoms in the system after processing:{traj_protein.n_atoms}")

    # traj_protein.save("traj_protein.dcd")  # 去溶剂后的轨迹
    # traj_protein.top.save_pdb("prmtop_protein.pdb")  # 新拓扑文件
    positions_stacked = []
    f, temp_path = tempfile.mkstemp()  # 创建临时文件路径 temp_path 用于保存单帧 PDB
    for i in tqdm.trange(0, len(traj_protein) // 1, 1):
        traj_protein[i].save_pdb(temp_path)  # 将当前帧保存为 PDB 文件

        with open(temp_path) as f:  # 读取 PDB 文件
            prot = protein.from_pdb_string(
                f.read()
            )  # 使用 protein.from_pdb_string 解析 PDB 文件
            pdb_feats = make_protein_features(prot, top_name)  # 生成蛋白质特征,特征字典
            positions_stacked.append(
                pdb_feats["all_atom_positions"]
            )  # 将当前帧的原子坐标添加到列表中

    pdb_feats["all_atom_positions"] = np.stack(
        positions_stacked
    )  # 将 positions_stacked 列表转换为 NumPy 数组
    np.savez(
        f"./trajectory/{pdb_name}/{config.traj_name}.npz", **pdb_feats
    )  # 保存特征字典为 .npz 文件
    os.unlink(temp_path)


if __name__ == "__main__":
    pdb_name = "1le3"
    # preprocess("traj", pdb_name)
    # MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF
    data = np.load(f"./trajectory/{pdb_name}/{config.traj_name}.npz", allow_pickle=True)
    # print(data["aatype"])  # (5000, num)
    # print(data.files)
    # 输出：['aatype', 'all_atom_positions', 'all_atom_mask','seq_length','sequence' ...]
    # resolution​​ (分辨率),​​domain_name​​ (结构域名称),between_segment_residues​​ (片段间残基),is_distillation​​ (是否为蒸馏数据)
    # 查看氨基酸类型 one-hot编码
    print("seq_length:", data["seq_length"][0])
    print("sequeue:", data["sequence"])  # 输出： (5000, num, 37, 3)

    # 查看最后一帧,残基的 CA 原子坐标
    # ca_pos = data["all_atom_positions"][-1, :, 1, :]  # 第0个残基，第1个原子（CA）
    # print(ca_pos)  # 示例：[10.2, 5.3, 8.7]
