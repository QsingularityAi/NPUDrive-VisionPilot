"""Homography projection between camera perspective and vehicle ground plane."""
import numpy as np
import cv2

def top_crop_2_1(height, width):
    """Calculates top crop coordinate to establish a 2:1 aspect ratio."""
    return max(0, int(round(float(height) - float(width) / 2.0)))

class HomographyTransformer:
    def __init__(self, H_raw, raw_shape=(1080, 1920), net_shape=(512, 1024)):
        self.H_raw = np.array(H_raw, dtype=np.float64)
        self.raw_h, self.raw_w = raw_shape[:2]
        self.net_h, self.net_w = net_shape[:2]
        
        # Calculate top-crop parameters
        self.crop_top = top_crop_2_1(self.raw_h, self.raw_w)
        self.crop_h = float(self.raw_h - self.crop_top)
        self.sx = float(self.raw_w) / float(self.net_w)
        self.sy = float(self.crop_h) / float(self.net_h)
        
        # Affine coordinate transformation T: model_input(1024x512) -> raw camera pixel
        self.T = np.array([
            [self.sx, 0.0,     0.0],
            [0.0,     self.sy, float(self.crop_top)],
            [0.0,     0.0,     1.0]
        ], dtype=np.float64)
        
        # H_resized maps model pixel coordinates directly to world coordinates
        self.H_resized = np.matmul(self.H_raw, self.T)
        self.H_world2resized = np.linalg.inv(self.H_resized)
        self.H_raw_inv = np.linalg.inv(self.H_raw)

    def img_to_world(self, u, v, is_model_space=True):
        """
        Projects image pixels (u, v) to vehicle ground plane (x_fwd, y_lat) in meters.
        x_fwd: forward distance (m)
        y_lat: lateral offset (m, +left, -right)
        """
        H = self.H_resized if is_model_space else self.H_raw
        pt = np.array([u, v, 1.0], dtype=np.float64)
        world_pt = np.matmul(H, pt)
        w = world_pt[2]
        if abs(w) < 1e-6:
            return 0.0, 0.0
        return world_pt[0] / w, world_pt[1] / w

    def world_to_img(self, x, y, to_model_space=True):
        """
        Projects vehicle ground coordinates (x_fwd, y_lat) in meters back to image pixel (u, v).
        """
        H_inv = self.H_world2resized if to_model_space else self.H_raw_inv
        pt = np.array([x, y, 1.0], dtype=np.float64)
        img_pt = np.matmul(H_inv, pt)
        w = img_pt[2]
        if abs(w) < 1e-6:
            return -1, -1
        u = img_pt[0] / w
        v = img_pt[1] / w
        return int(round(u)), int(round(v))
