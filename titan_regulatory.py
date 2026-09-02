"""
================================================================================
STARTNERVE INTELLIGENCE — IN-SILICO REGULATORY AUDIT REPORT GENERATOR
================================================================================
Function: Ingests a candidate SMILES string, evaluates structural risks via 
          Titan V11 parameters, and compiles an automated PDF assessment.
================================================================================
"""

import os
import sys
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path

# Bring in architecture variables from your clean training script
from train_v11_titan import TitanV11, SCHNET_CUTOFF

class TitanInferenceEngine:
    def __init__(self, checkpoint_path="titan_checkpoints/best.pt"):
        self.device = torch.device("cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self.tasks = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
            "NR-PPAR-γ", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]
        self.model = self._load_model()

    def _load_model(self):
        """Initializes the network and binds the optimized trained weights."""
        if not self.checkpoint_path.exists():
            print(f"⚠️ Warning: Checkpoint '{self.checkpoint_path}' not found yet. Using initialized weights.")
            model = TitanV11(node_feat_dim=node_feat_dim)
            model.eval()
            return model
            
        print(f"🛰️  Binding trained weight matrices from {self.checkpoint_path} into inference layers...")
        model = TitanV11(node_feat_dim=node_feat_dim)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def featurize_smiles(self, smiles: str):
        """Converts a live query molecule string into true 3D graph tensors on the fly."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid chemical SMILES token.")
                
            # Mirror the exact V1 fallback featurization logic from your stable data loaders
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=42)
            conformer = mol.GetConformer()
            
            # Extract structural topologies
            z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
            num_nodes = z.shape[0]
            
            # Upscale element mapping cleanly to 162-dim channel bounds
            x = torch.zeros((num_nodes, 162), dtype=torch.float)
            for i in range(num_nodes):
                atomic_num = int(z[i].item())
                if 1 <= atomic_num <= 118:
                    x[i, atomic_num - 1] = 1.0
                    
            pos = torch.tensor([list(conformer.GetAtomPosition(i)) for i in range(num_nodes)], dtype=torch.float)
            
            edge_indices = []
            for bond in mol.GetBonds():
                start_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                edge_indices.append([start_idx, end_idx])
                edge_indices.append([end_idx, start_idx])
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
            
            # Mock batch array tracking index to map cleanly into PyG global pooling layers
            batch = torch.zeros(num_nodes, dtype=torch.long)
            
            # Package into an inference-safe mock class object structure
            class InferenceBatch:
                def __init__(self, x, z, pos, edge_index, batch):
                    self.x = x
                    self.z = z
                    self.pos = pos
                    self.edge_index = edge_index
                    self.batch = batch
                def to(self, device):
                    self.x = self.x.to(device)
                    self.z = self.z.to(device)
                    self.pos = self.pos.to(device)
                    self.edge_index = self.edge_index.to(device)
                    self.batch = self.batch.to(device)
                    return self
                    
            return InferenceBatch(x, z, pos, edge_index, batch)
        except Exception as e:
            print(f"❌ Structural featurization pipeline broken for query molecule: {e}")
            return None

    def audit_molecule(self, smiles: str):
        """Passes a single compound tensor stream completely through GNN layers."""
        batch_data = self.featurize_smiles(smiles)
        if batch_data is None:
            return None
            
        with torch.no_grad():
            batch_data = batch_data.to(self.device)
            logits = self.model(batch_data)
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
            
        audit_results = {}
        for idx, task_name in enumerate(self.tasks):
            audit_results[task_name] = float(probabilities[idx])
        return audit_results

    def generate_pdf_report(self, smiles: str, output_pdf="regulatory_audit_report.pdf"):
        """Evaluates structural liabilities and structures properties text."""
        print(f"\n🚀 Initiating In-Silico screening audit for structure target: {smiles}")
        raw_predictions = self.audit_molecule(smiles)
        if not raw_predictions:
            print("❌ Audit execution aborted due to upstream tracking errors.")
            return

        print("\n" + "="*60)
        print("          STARTNERVE INTELLIGENCE COMPLIANCE RESULTS          ")
        print("="*60)