from typing import Dict, List, Union

import cv2.xfeatures2d
import numpy as np

from .dataloader import Dataloader


class Model:
    def __init__(self, templates: Dataloader, significance: float = 0.85, distance_factor: float = 0.85) -> None:
        self.templates = templates
        self.significance = significance
        self.distance_factor = distance_factor
        self._sift = cv2.SIFT_create()
        self._flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 3}, {'checks': 10})

    def __call__(self, img: Union[np.ndarray, Dataloader], *args, **kwargs) \
            -> Union[Dict[str, Dict[str, bool]], Dict[str, bool]]:
        if isinstance(img, np.ndarray):
            return self._infer_image(img[0])
        elif isinstance(img, Dataloader):
            return self._infer_batch(img)
        else:
            raise TypeError

    def _infer_batch(self, images: List[np.ndarray]) -> Dict[str, Dict[str, bool]]:
        result = dict()
        for i, (img, name) in enumerate(images):
            result[name] = self._infer_image(img)
        return result

    def _infer_image(self, img: List[np.ndarray]) -> Dict[str, Dict[str, bool]]:
        result = dict()
        kp_img, des_img = self._sift.detectAndCompute(img, None)
        for template, name in self.templates:
            kp_t, des_t = self._sift.detectAndCompute(template, None)
            matches = self._flann.knnMatch(des_img, des_t, k=2)
            good_matches = [m for m, n in matches if m.distance < self.distance_factor * n.distance]
            result[name.split('.')[0]] = True if len(good_matches) >= self.significance * len(des_t) else False
        return result
