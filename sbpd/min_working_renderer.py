import sys, os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # add Visual-Navigation-Release to PYTHONPATH
os.environ['PYOPENGL_PLATFORM'] = 'egl' # we have to explicitly choose the opengl platform on linux

import sbpd.sbpd_renderer as sbpdr
import params.renderer_params as rp

params = rp.create_params()
renderer = sbpdr.SBPDRenderer(params)