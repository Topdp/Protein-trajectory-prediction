# utils.py
import os
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
from util.chemical import RESIDUE_TYPES, ATOM_TYPES, ATOM_ELEMENTS


def create_pdb_structure_simple(
    pred_coords, residue_types, atom_mask, frame_idx, output_path
):
    n_res = len(residue_types)

    # 创建结构
    structure = Structure.Structure(f"frame_{frame_idx}")
    model = Model.Model(0)
    chain = Chain.Chain("A")

    atom_counter = 1
    for res_idx in range(n_res):
        res_type_idx = residue_types[res_idx].item()
        res_type = RESIDUE_TYPES[res_type_idx]

        residue = Residue.Residue((" ", res_idx + 1, " "), res_type, "")

        for atom_idx in range(14):
            if atom_mask[res_idx, atom_idx] > 0.5:
                atom_type = ATOM_TYPES[atom_idx]
                element = ATOM_ELEMENTS[atom_type]
                atom_coord = pred_coords[res_idx, atom_idx]

                atom = Atom.Atom(
                    name=atom_type,
                    coord=atom_coord,
                    bfactor=0.0,
                    occupancy=1.0,
                    altloc=" ",
                    fullname=atom_type.ljust(4),
                    serial_number=atom_counter,
                    element=element,
                )
                residue.add(atom)
                atom_counter += 1

        chain.add(residue)

    model.add(chain)
    structure.add(model)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path)
    print(f"保存重建结构到: {output_path}")
