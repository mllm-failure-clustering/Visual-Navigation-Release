import os, sys
sys.path.append("/home/leo/git/Visual-Navigation-Release")
os.system('export PYOPENGL_PLATFORM=egl')   # we have to explicitly choose the platform

import sbpd.sbpd_renderer as sbpdr
import params.renderer_params as rp


params = rp.create_params()
renderer = sbpdr.SBPDRenderer(params)