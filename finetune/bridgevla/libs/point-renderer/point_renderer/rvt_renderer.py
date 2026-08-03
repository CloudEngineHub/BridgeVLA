import torch
import point_renderer.ops as ops
from point_renderer.cameras import OrthographicCameras, PerspectiveCameras
from point_renderer.renderer import PointRenderer

import point_renderer.rvt_ops as rvt_ops


class RVTBoxRenderer():
    """
    Wrapper around PointRenderer that fixes the cameras to be orthographic cameras
    on the faces of a 2x2x2 cube placed at the origin
    """

    def __init__(
        self,
        img_size,
        radius=None,
        default_color=0.0,
        default_depth=-1.0,
        antialiasing_factor=1,
        pers=False,
        normalize_output=True,
        with_depth=True,
        device="cuda",
        perf_timer=False,
        strict_input_device=True,
        no_down=True,
        no_top=False,
        three_views=False,
        two_views=False,
        one_view=False,
        oblique_views=False,
        add_3p=False,
        flip_top_up=False,
        rotate_top_ccw90=False,
        img_sizes_w=None,
        **kwargs):

        self.renderer = PointRenderer(device=device, perf_timer=perf_timer)

        self.img_size = img_size
        # splat_radius is in world units; for an orthographic camera the kernel radius = ceil(fx * radius) pixels.
        # With img_sizes_w=2 (the default) fx = img_size_px/2: 224/2 = 112, so
        #   0.012 → ceil(112*0.012)=2 radius → a 5x5 kernel (matching the original BridgeVLA).
        # When the caller passes no radius:
        #   - oblique defaults to 2.0 (pulled in, same fx as axis) → 0.012 already gives 5x5, matching axis;
        #   - other view combinations still default to 0.012.
        if radius is None:
            radius = 0.012
        self.splat_radius = radius
        self.default_color = default_color
        self.default_depth = default_depth
        self.aa_factor = antialiasing_factor
        self.normalize_output = normalize_output
        self.with_depth = with_depth

        self.strict_input_device = strict_input_device

        # oblique_views implies 3 views and is mutually exclusive with two/one view.
        assert not (oblique_views and (two_views or one_view)), \
            "oblique_views only supports the 3-view configuration"
        self.oblique_views = oblique_views

        # Pre-compute fixed cameras ahead of time
        self.cameras = self._get_cube_cameras(
            img_size=self.img_size,
            orthographic=not pers,
            no_down=no_down,
            no_top=no_top,
            three_views=three_views,
            two_views=two_views,
            one_view=one_view,
            oblique_views=oblique_views,
            img_sizes_w=img_sizes_w,
            add_3p=add_3p,
            flip_top_up=flip_top_up,
            rotate_top_ccw90=rotate_top_ccw90,
        )
        self.cameras = self.cameras.to(device)

        # TODO(Valts): add support for dynamic cameras

        # Cache
        self._fix_pts_cam = None
        self._fix_pts_cam_wei = None
        self._pts = None

        # RVT API (that we might want to refactor)
        self.num_img = len(self.cameras)
        self.only_dyn_cam = False

    def _check_device(self, input, input_name):
        if self.strict_input_device:
            assert str(input.device) == str(self.renderer.device), (
                f"Input {input_name} (device {input.device}) should be on the same device as the renderer ({self.renderer.device})")

    def _get_cube_cameras(
        self,
        img_size,
        orthographic,
        no_down,
        no_top,
        three_views,
        two_views,
        one_view,
        oblique_views,
        add_3p,
        flip_top_up=False,
        rotate_top_ccw90=False,
        img_sizes_w=None,
    ):
        top_up = [0, -1, 0] if flip_top_up else [0, 1, 0]
        if rotate_top_ccw90:
            # CCW 90° about +Z: (x, y) -> (-y, x); equivalent to rotating the
            # rendered top image CW 90° on screen.
            top_up = [-top_up[1], top_up[0], top_up[2]]
        # Oblique-view constants (a = sqrt(2/3), b = 1/sqrt(3)):
        #   the angle between eye_i and +Z = arccos(1/sqrt 3) ≈ 54.7356°, azimuths 0/120/240.
        #   up_i = normalize(z_hat - (z_hat·eye_i) * eye_i), i.e. the projection of world +Z onto the image plane.
        _a = 0.81649658092772603  # sqrt(2/3)
        _b = 0.57735026918962573  # 1/sqrt(3)
        cam_dict = {
            "top": {"eye": [0, 0, 1], "at": [0, 0, 0], "up": top_up},
            "front": {"eye": [1, 0, 0], "at": [0, 0, 0], "up": [0, 0, 1]},
            "down": {"eye": [0, 0, -1], "at": [0, 0, 0], "up": [0, 1, 0]},
            "back": {"eye": [-1, 0, 0], "at": [0, 0, 0], "up": [0, 0, 1]},
            "left": {"eye": [0, -1, 0], "at": [0, 0, 0], "up": [0, 0, 1]},
            # eye=[0, 1, 0] (NOT [0, 0.5, 0]): the camera must sit on the y=+1
            # face of the cube, symmetric with "left" at y=-1. The original
            # initial-commit value [0, 0.5, 0] put the camera INSIDE the [-1,1]^3
            # cube, so points with cube-space y > 0.5 had negative depth in the
            # camera frame and were drawn on top of legitimate front points
            # (z-buffer pollution). Visible as tilted/floating streaks in the
            # right-view dumps for any scene with non-trivial spread along y.
            "right": {"eye": [0, 1, 0], "at": [0, 0, 0], "up": [0, 0, 1]},
            "oblique_a": {"eye": [ _a,          0.0,         _b],
                          "at":  [0, 0, 0],
                          "up":  [-_b,          0.0,         _a]},
            "oblique_b": {"eye": [-_a / 2.0,    _a * 0.86602540378443865, _b],
                          "at":  [0, 0, 0],
                          "up":  [ _b / 2.0,   -0.5,         _a]},
            "oblique_c": {"eye": [-_a / 2.0,   -_a * 0.86602540378443865, _b],
                          "at":  [0, 0, 0],
                          "up":  [ _b / 2.0,    0.5,         _a]},
        }

        assert not (two_views and three_views)
        assert not (one_view and three_views)
        assert not (one_view and two_views)
        assert not (oblique_views and (two_views or one_view))
        assert not add_3p, "Not supported with point renderer yet,"
        if two_views or three_views or one_view or oblique_views:
            if no_down or no_top or add_3p:
                print(
                    f"WARNING: when three_views={three_views} or two_views={two_views} "
                    f"or oblique_views={oblique_views} -- "
                    f"no_down={no_down} no_top={no_top} add_3p={add_3p} does not matter."
                )

        if oblique_views:
            cam_names = ["oblique_a", "oblique_b", "oblique_c"]
        elif three_views:
            cam_names = ["top", "front", "right"]
        elif two_views:
            cam_names = ["top", "front"]
        elif one_view:
            cam_names = ["front"]
        else:
            cam_names = ["top", "front", "down", "back", "left", "right"]
            if no_down:
                # select index of "down" camera and remove it from the list
                del cam_names[cam_names.index("down")]
            if no_top:
                del cam_names[cam_names.index("top")]


        cam_list = [cam_dict[n] for n in cam_names]
        eyes = [c["eye"] for c in cam_list]
        ats = [c["at"] for c in cam_list]
        ups = [c["up"] for c in cam_list]

        if orthographic:
            # img_sizes_w is the camera's field of view as physical width/height in world coordinates.
            #  - [2, 2] makes the axis-aligned cameras cover exactly [-1,1]^3 (matching the original BridgeVLA).
            #  - Under oblique views the largest diagonal of [-1,1]^3 projects to ~3.21, but place_pc_in_cube
            #    squeezes the actual point cloud into ~[-0.8, 0.8], so [2.0, 2.0] is still the default and pulls
            #    the view in — fx is then on par with axis, and a 0.012 radius gives exactly a 5x5 kernel, i.e.
            #    the same granularity as axis. Extreme corner points (further than 1.0 from the origin) would
            #    fall outside the frame, but essentially none survive place_pc_in_cube.
            # Pass `img_sizes_w` explicitly to override the default (when a wider field of view is needed).
            if img_sizes_w is None:
                img_sizes_w = [2.0, 2.0]
            cameras = OrthographicCameras.from_lookat(eyes, ats, ups, img_sizes_w=img_sizes_w, img_size_px=img_size)
        else:
            cameras = PerspectiveCameras.from_lookat(eyes, ats, ups, hfov=70, img_size=img_size)
        return cameras

    @torch.no_grad()
    def get_pt_loc_on_img(self, pt, fix_cam=False, dyn_cam_info=None):
        """
        returns the location of a point on the image of the cameras
        :param pt: torch.Tensor of shape (bs, np, 3)
        :returns: the location of the pt on the image. this is different from the
            camera screen coordinate system in pytorch3d. the difference is that
            pytorch3d camera screen projects the point to [0, 0] to [H, W]; while the
            index on the img is from [0, 0] to [H-1, W-1]. We verified that
            the to transform from pytorch3d camera screen point to img we have to
            subtract (1/H, 1/W) from the pytorch3d camera screen coordinate.
        :return type: torch.Tensor of shape (bs, np, self.num_img, 2)
        """
        assert len(pt.shape) == 3
        assert pt.shape[-1] == 3
        assert fix_cam, "Not supported with point renderer"
        assert dyn_cam_info is None, "Not supported with point renderer"

        bs, np, _ = pt.shape

        self._check_device(pt, "pt")

        # TODO(Valts): Ask Ankit what what is the bs dimension here, and treat it correctly here

        pcs_px = []
        for i in range(bs):
            pc_px, pc_cam = ops.project_points_3d_to_pixels(
                pt[i], self.cameras.inv_poses, self.cameras.intrinsics, self.cameras.is_orthographic())
            pcs_px.append(pc_px)
        pcs_px = torch.stack(pcs_px, dim=0)
        pcs_px = torch.permute(pcs_px, (0, 2, 1, 3))

        # TODO(Valts): Double-check with Ankit that these projections are truly pixel-aligned
        return pcs_px

    @torch.no_grad()
    def get_feat_frm_hm_cube(self, hm, fix_cam=False, dyn_cam_info=None):
        """
        :param hm: torch.Tensor of (1, num_img, h, w)
        :return: tupe of ((num_img, h^3, 1), (h^3, 3))
        """
        x, nc, h, w = hm.shape
        assert x == 1
        assert nc == self.num_img
        assert self.img_size == (h, w)
        assert fix_cam, "Not supported with point renderer"
        assert dyn_cam_info is None, "Not supported with point renderer"

        self._check_device(hm, "hm")

        if self._pts is None:
            res = self.img_size[0]
            pts = torch.linspace(-1 + (1 / res), 1 - (1 / res), res).to(hm.device)
            pts = torch.cartesian_prod(pts, pts, pts)
            self._pts = pts

        pts_hm = []

        # if self._fix_cam
        if self._fix_pts_cam is None:
            # (np, nc, 2)
            pts_img = self.get_pt_loc_on_img(self._pts.unsqueeze(0),
                                             fix_cam=True).squeeze(0)
            # pts_img = pts_img.permute((1, 0, 2))
            # (nc, np, bs)
            fix_pts_hm, pts_cam, pts_cam_wei = rvt_ops.select_feat_from_hm(
                pts_img.transpose(0, 1), hm.transpose(0, 1)[0 : len(self.cameras)]
            )
            self._fix_pts_img = pts_img
            self._fix_pts_cam = pts_cam
            self._fix_pts_cam_wei = pts_cam_wei
        else:
            pts_cam = self._fix_pts_cam
            pts_cam_wei = self._fix_pts_cam_wei
            fix_pts_hm = rvt_ops.select_feat_from_hm_cache(
                pts_cam, hm.transpose(0, 1)[0 : len(self.cameras)], pts_cam_wei
            )
        pts_hm.append(fix_pts_hm)

        #if not dyn_cam_info is None:
        # TODO(Valts): implement
        pts_hm = torch.cat(pts_hm, 0)
        return pts_hm, self._pts

    @torch.no_grad()
    def get_max_3d_frm_hm_cube(self, hm, fix_cam=False, dyn_cam_info=None,
                               topk=1, non_max_sup=False,
                               non_max_sup_dist=0.02):
        """
        given set of heat maps, return the 3d location of the point with the
            largest score, assumes the points are in a cube [-1, 1]. This function
            should be used  along with the render. For standalone version look for
            the other function with same name in the file.
        :param hm: (1, nc, h, w)
        :return: (1, topk, 3)
        """
        assert fix_cam, "Not supported with point renderer"
        assert dyn_cam_info is None, "Not supported with point renderer"

        self._check_device(hm, "hm")

        x, nc, h, w = hm.shape
        assert x == 1
        assert nc == len(self.cameras)
        assert self.img_size == (h, w)

        pts_hm, pts = self.get_feat_frm_hm_cube(hm, fix_cam, dyn_cam_info)
        # (bs, np, nc)
        pts_hm = pts_hm.permute(2, 1, 0)
        # (bs, np)
        pts_hm = torch.mean(pts_hm, -1)
        if non_max_sup and topk > 1:
            _pts = pts.clone()
            pts = []
            pts_hm = torch.squeeze(pts_hm, 0)
            for i in range(topk):
                ind_max_pts = torch.argmax(pts_hm, -1)
                sel_pts = _pts[ind_max_pts]
                pts.append(sel_pts)
                dist = torch.sqrt(torch.sum((_pts - sel_pts) ** 2, -1))
                pts_hm[dist < non_max_sup_dist] = -1
            pts = torch.stack(pts, 0).unsqueeze(0)
        else:
            ind_max_pts = torch.topk(pts_hm, topk)[1]
            # (bs, topk, 3)
            pts = pts[ind_max_pts]
        return pts

    def __call__(self, pc, feat, fix_cam=False, dyn_cam_info=None):

        self._check_device(pc, "pc")
        self._check_device(pc, "feat")

        pc_images, pc_depths = self.renderer.render_batch(pc, feat,
                                    cameras=self.cameras,
                                    img_size=self.img_size,
                                    splat_radius=self.splat_radius,
                                    default_color=self.default_color,
                                    default_depth=self.default_depth,
                                    aa_factor=self.aa_factor
                                    )

        if self.normalize_output:
            _, h, w = pc_depths.shape
            depth_0 = pc_depths == -1
            depth_sum = torch.sum(pc_depths, (1, 2)) + torch.sum(depth_0, (1, 2))
            depth_mean = depth_sum / ((h * w) - torch.sum(depth_0, (1, 2)))
            pc_depths -= depth_mean.unsqueeze(-1).unsqueeze(-1)
            pc_depths[depth_0] = -1

        if self.with_depth:
            img_out = torch.cat([pc_images, pc_depths[:, :, :, None]], dim=-1)
        else:
            img_out = pc_images
        # For Debug
        # import numpy as np
        # import cv2
        # import time
        # import os
        # init_folder=time.time()
        # for idx in range(pc_images.shape[0]):
        #     debug_img = (255 * pc_images[idx, :, :, 3:6].cpu().numpy()).astype(np.uint8)
        #     # debug_img = (255 * pc_images[idx, :, :, :3].cpu().numpy()).astype(np.uint8)
        #     # debug_img = Image.fromarray(debug_img)
        #     # debug_img.save(f'debug_img/{time.time()}__{self.task_name}_{idx}.png')
        #     if not os.path.exists(f'debug_img/{init_folder}'):
        #         os.makedirs(f'debug_img/{init_folder}')
        #     cv2.imwrite(f'debug_img/{init_folder}/{idx}.png', debug_img)
            
        return img_out
