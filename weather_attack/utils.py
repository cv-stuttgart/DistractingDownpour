import torch
import numpy as np


def put_things_on_img(img, points, what):
    assert len(img.shape) == 3
    assert img.shape[-1] == 3
    assert len(points.shape) == 2
    assert points.shape[1] == 2
    assert len(what.shape) == 2
    assert what.shape[1] == 3
    assert what.shape[0] == points.shape[0]

    h,w,_ = img.shape
    indices = points[:,0] + w * points[:,1]
    indices = indices.unsqueeze(1).repeat(1,3)
    img_flat = img.reshape(-1,3)
    img_flat.scatter_(0, indices, what)
    img2 = img_flat.reshape(h,w,3)
    return img2


def save_weatherfile(file, weather):
  points3D, motion3D, flakes, flakescol, flakestrans = weather
  np.savez(file, points3D=points3D.numpy(), motion3D=motion3D.numpy(), flakes=flakes.numpy(), flakescol=flakescol.numpy(), flakestrans=flakestrans.numpy())


def load_weather(path):
    try:
      weather = np.load(path, allow_pickle=True)
      points3D = weather["points3D"]
      motion3D = weather["motion3D"]
      flakes = weather["flakes"]
      flakescol = weather["flakescol"]
      flakestrans = weather["flakestrans"]
      weather = (points3D, motion3D, flakes, flakescol, flakestrans)
      success = True
    except FileNotFoundError as e:
      print(f"Failed to load weather data from {path}, File Not Found. Please generate weather instead.")
      weather = None
      success = False
    return weather, success

