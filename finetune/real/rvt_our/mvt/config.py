# Adapted from BridgeVLA/finetune/bridgevla/mvt/config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.depth = 8
_C.img_size = 220
_C.img_feat_dim = 3
_C.feat_dim = (72 * 3) + 2 + 2
_C.im_channels = 64
_C.activation = "lrelu"
_C.decoder_dropout = 0.0
_C.img_patch_size = 11
_C.final_dim = 64
_C.self_cross_ver = 1
_C.add_corr = True
_C.norm_corr = False
_C.add_pixel_loc = True
_C.add_depth = True
_C.rend_three_views = False
# True: three mutually perpendicular oblique orthographic cameras looking down (magic angle 54.7356°,
# azimuths 0/120/240). Implies 3 views; mutually exclusive with rend_three_views' axis-aligned top/front/right.
_C.rend_oblique_views = False
_C.use_point_renderer = False
_C.pe_fix = True
_C.feat_ver = 0
_C.wpt_img_aug = 0.01
_C.inp_pre_pro = True
_C.inp_pre_con = True
_C.cvx_up = False
_C.xops = False
_C.rot_ver = 0
_C.num_rot = 72
_C.stage_two = False
_C.st_sca = 4
_C.st_wpt_loc_aug = 0.05
_C.st_wpt_loc_inp_no_noise = False
_C.img_aug_2 = 0.0


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()
