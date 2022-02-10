from typing import Iterable, Tuple

from os import listdir
from os.path import join

import cv2
import numpy as np


class Dataloader:
    def __init__(self, directory: str = 'data') -> None:
        self.directory = directory
        self.files = [file for file in listdir(self.directory) if file[-3:] == 'jpg' or file[-4:] == 'jpeg']

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, key) -> Tuple[np.ndarray, str]:
        if key > len(self):
            raise KeyError
        else:
            file_name = self.files[key]
            img = self._read_img(file_name)
            return img, file_name

    def __iter__(self) -> Iterable:
        self._idx = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, str]:
        if self._idx < len(self):
            img, file_name = self[self._idx]
            self._idx += 1
            return img, file_name
        else:
            raise StopIteration

    def _read_img(self, file_name: str):
        file_path = join(self.directory, file_name)
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img


class CachedDataloader(Dataloader):
    def __init__(self, directory: str = 'data') -> None:
        super().__init__(directory)
        self._cache = dict()

    def __getitem__(self, key) -> Tuple[np.ndarray, str]:
        if key > len(self):
            raise KeyError
        else:
            file_name = self.files[key]
            if file_name in self._cache.keys():
                img = self._cache[file_name]
            else:
                img = self._read_img(file_name)
                self._cache[file_name] = img
            return img, file_name
