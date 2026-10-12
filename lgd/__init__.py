def _patch_yacs_utf8(): #Windows defaults to GBK and fails on yaml holding non-ASCII chars; always open as utf-8
    try:
        from yacs.config import CfgNode
    except ImportError:
        return
    if getattr(CfgNode.merge_from_file, '_utf8_patched', False): return
    def merge_from_file(self, cfg_filename):
        with open(cfg_filename, 'r', encoding='utf-8') as f: self.merge_from_other_cfg(self.load_cfg(f.read()))
    merge_from_file._utf8_patched = True; CfgNode.merge_from_file = merge_from_file

_patch_yacs_utf8()
from .config import *
from .encoder import *
from .head import *
from .layer import *
from .loader import *
from .loss import *
from .model import *
from .optimizer import *
from .train import *
from .transform import *
from .ddpm import *
