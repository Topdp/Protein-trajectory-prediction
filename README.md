# PTGMamba: Protein-trajectory-prediction
The ongoing project is based on the EGNN and Mamba models, aiming to predict the molecular dynamics trajectories (MD) of proteins, using GNN to map all the structures in the protein trajectories, embed them with features and obtain pooled vectors, and input them into Mamba to learn to predict future frames through the Selective State Space Model (SSM)

## Model Architecture


### Model used
- E(n) Equivariant Graph Neural Networks (EGNN)
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Mamba)
- Attention Mechanism

### Deployment and use
You need to prepare the protein pdb file, trajectory file, and topology file, and place them in the trajectory folder for later use.

#### train
--use_cache： You may add --use_cache. This means that you will be using the processed trajectory cache file. Default not in use.
```
python ./main/run.py  --p_Name 2ala --top_Name 2ala --traj_name traj --window_size 10 --batch_size 16 --epochs 120 --lr 1e-4 --dropout 0.5 --d_model 256 --d_state 64 --n_layers 4 --depth 4 --dim 256 --edge_dim 64
```
More options are presented in the run file.

### Version Control
This project uses Git for version management. You can view the currently available versions in the repository.
