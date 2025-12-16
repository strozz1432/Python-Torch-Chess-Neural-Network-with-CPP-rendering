import os
import pickle
from typing import Dict

import torch
import torch.nn as nn

from .model import ChessModel


class OnlineTrainer:
    def __init__(self, model_path: str = None, mapping_path: str = None, lr: float = 1e-4, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.lr = lr

        cwd = os.getcwd()
        self.mapping_path = mapping_path or os.path.join(cwd, "models", "move_to_int")
        self.model_path = model_path or os.path.join(cwd, "models", "TORCH_ONLINE.pth")

        self.move_to_int: Dict[str, int] = {}
        if os.path.exists(self.mapping_path):
            try:
                with open(self.mapping_path, "rb") as f:
                    self.move_to_int = pickle.load(f)
            except Exception:
                self.move_to_int = {}

        num_classes = max(1, len(self.move_to_int))
        self.model = ChessModel(num_classes=num_classes).to(self.device)

        if os.path.exists(self.model_path):
            try:
                state = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state)
            except Exception:
                pass

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def _save_mapping(self):
        os.makedirs(os.path.dirname(self.mapping_path), exist_ok=True)
        with open(self.mapping_path, "wb") as f:
            pickle.dump(self.move_to_int, f)

    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)

    def _ensure_move_index(self, move_uci: str) -> int:
        if move_uci in self.move_to_int:
            return self.move_to_int[move_uci]
        # New index
        new_index = len(self.move_to_int)
        self.move_to_int[move_uci] = new_index

        # Expand layer
        old_fc = self.model.fc2
        old_out = old_fc.out_features
        in_features = old_fc.in_features

        new_out = old_out + 1
        new_fc = nn.Linear(in_features, new_out)

        # Copy weights
        with torch.no_grad():
            new_fc.weight[:old_out].data.copy_(old_fc.weight.data)
            new_fc.bias[:old_out].data.copy_(old_fc.bias.data)
            nn.init.xavier_uniform_(new_fc.weight[old_out:].unsqueeze(0))

        self.model.fc2 = new_fc.to(self.device)
        # Recreate opt
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Save map
        try:
            self._save_mapping()
        except Exception:
            pass

        return new_index

    def update(self, board_matrix, move_uci: str, merits_delta: float = 0.0):
        import numpy as _np

        if move_uci is None:
            return

        idx = self._ensure_move_index(move_uci)

        x = torch.tensor(_np.array(board_matrix, dtype=_np.float32))[None].to(self.device)
        # NCHW input
        logits = self.model(x)

        target = torch.tensor([idx], dtype=torch.long, device=self.device)

        losses = self.criterion(logits, target)

        # Weight scale:
        weight = max(0.1, 1.0 + (merits_delta or 0.0) / 5.0)
        loss = (losses * weight).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Logging
        try:
            print(f"trainer update: move={move_uci} idx={idx} merits={merits_delta} w={weight:.3f} loss={loss.item():.6f}")
        except Exception:
            pass

        # Save model
        try:
            self._save_model()
        except Exception:
            pass
