from pyviewer.toolbar_viewer import AutoUIViewer
from pyviewer.params import *
from pyviewer.utils import reshape_grid
from copy import deepcopy
import numpy as np
import torch
import yaml
from tqdm import tqdm
from inference.utils import downscale_images
from train import WurstCoreB

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Inputs based on which result is cached
@strict_dataclass
class State(ParamContainer):
    B_steps: Param = IntParam('B steps', 5, 2, 20)
    B: Param = IntParam('Batch size', 1, 1, 9)
    factor: Param = EnumSliderParam('Downscale factor', '0.75', 
        ['0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85', '0.90', '0.95', '1.0'])
    cfg: Param = EnumSliderParam('cfg', None,
        [0.5, 0.9, None, 1.1, 1.5, 1.9, 2.3, 2.7, 3.1])
    use_capt: Param = BoolParam('Use caption', False)
    seed: Param = IntParam('Seed', 0, 0, 999, buttons=True)
    img_idx: Param = IntParam('Image idx', 0, 0, 63, buttons=True)
    show_ref: Param = BoolParam('Show reference', True)

class Viewer(AutoUIViewer):
    def setup_state(self):
        self.state = State()
        self.init()
        self.cache = dict()
        self.dset_elems = [] # enable random access

    def init(self):
        config_file_b = 'configs/inference/stage_b_3b.yaml'
        with open(config_file_b, "r", encoding="utf-8") as file:
            config_file_b = yaml.safe_load(file)
        config_file_b['batch_size'] = 1 # want dataloader with batch size 1
        self.core = WurstCoreB(config_dict=config_file_b, device=device, training=False)
        self.extras = self.core.setup_extras_pre()
        self.data = self.core.setup_data(self.extras, dset_workers=0)
        self.iter = iter(self.data.dataloader)
        self.models = self.core.setup_models(self.extras)
        self.models.generator.bfloat16()

    def get_batch(self, idx, B):
        while len(self.dset_elems) < idx + B:
            self.dset_elems.append(next(self.iter))
        elems = self.dset_elems[idx:idx+B]
        return {
            'images': torch.cat([e['images'] for e in elems]),
            'captions': sum([e['captions'] for e in elems], [])
        }

    def compute(self):
        key = str(self.state)
        if key in self.cache:
            return self.cache[key]
        ret = self.process(deepcopy(self.state))
        if ret is not None: # not aborted
            self.cache[key] = ret
        return ret

    def process(self, state):
        batch = self.get_batch(self.state.img_idx, self.state.B)

        if state.show_ref:
            return reshape_grid(batch['images'])

        # Stage B Parameters
        self.extras.sampling_configs['cfg'] = state.cfg
        self.extras.sampling_configs['shift'] = 1
        self.extras.sampling_configs['timesteps'] = state.B_steps
        self.extras.sampling_configs['t_start'] = 1.0

        # Make copy of batch without captions
        batch_no_capt = { k: v for k,v in batch.items() }
        batch_no_capt['captions'] = ['']*len(batch['captions'])

        # Include or drop text conditioning.
        # Barely any effect on output!
        cond_batch = batch if state.use_capt else batch_no_capt

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            torch.manual_seed(state.seed)

            print("Original Size:", cond_batch['images'].shape)
            factor = float(state.factor)
            scaled_image = downscale_images(cond_batch['images'], factor)
            print("Downscaled Size:", scaled_image.shape)
            
            effnet_latents = self.models.effnet(self.extras.effnet_preprocess(scaled_image.to(device)))
            print("Encoded Size:", effnet_latents.shape)
            
            conditions = self.core.get_conditions(cond_batch, self.models, self.extras, is_eval=True, is_unconditional=False)
            unconditions = self.core.get_conditions(cond_batch, self.models, self.extras, is_eval=True, is_unconditional=True)
            conditions['effnet'] = effnet_latents # overwrite fullres with scaled
            unconditions['effnet'] = torch.zeros_like(effnet_latents)

            sampling_b = self.extras.gdf.sample(
                self.models.generator, conditions, (cond_batch['images'].size(0), 4, cond_batch['images'].size(-2)//4, cond_batch['images'].size(-1)//4),
                unconditions, device=device, **self.extras.sampling_configs
            )
            for (sampled_b, _, _) in tqdm(sampling_b, total=self.extras.sampling_configs['timesteps']):
                if self.state != state:
                    return None # abort

            sampled = self.models.stage_a.decode(sampled_b).float()
            print("Decoded Size:", sampled.shape)
            print("")

        return reshape_grid(sampled)

viewer = Viewer('SISR viz')