"""CPU-only saved-network inference; never imports training or fits any parameters."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from daily.neural_architecture import SharedForecaster
from daily.neural_features import CHANNELS


class NeuralPredictor:
    def __init__(self,path):
        path=Path(path);self.bundle=json.loads(path.read_text(encoding='utf-8'));b=self.bundle
        if b.get('task')!='shared_direct_neural_return' or b.get('schema_version')!=1:raise ValueError('Unknown neural model')
        if b['channels']!=list(CHANNELS) or b['horizons']!=[1,5,21,63,126]:raise ValueError('Neural feature schema mismatch')
        architecture=Path(__file__).with_name('neural_architecture.py')
        if hashlib.sha256(architecture.read_bytes()).hexdigest()!=b['metadata']['source_sha256']['code/research/neural_models.py']:
            raise ValueError('Network architecture differs from training')
        name=b['weights']['file']
        if Path(name).name!=name:raise ValueError('Invalid neural weight filename')
        native=path.parent/name
        if hashlib.sha256(native.read_bytes()).hexdigest()!=b['weights']['sha256']:raise ValueError('Neural weight checksum mismatch')
        torch.set_num_threads(2)
        self.net=SharedForecaster(b['architecture'],b['sequence_length'],len(b['channels']),len(b['codes']),len(b['horizons']))
        self.net.load_state_dict(load_file(str(native),device='cpu'),strict=True);self.net.eval()

    def predict(self,x,identities):
        x=np.asarray(x,dtype=np.float32)
        if not np.isfinite(x).all():raise ValueError('Nonfinite neural features')
        with torch.inference_mode():out=self.net(torch.from_numpy(x),torch.as_tensor(identities,dtype=torch.long)).numpy()
        values=out.astype(np.float64)*np.asarray(self.bundle['target_scales'],dtype=np.float64)
        if not np.isfinite(values).all():raise ValueError('Nonfinite neural output')
        return values
