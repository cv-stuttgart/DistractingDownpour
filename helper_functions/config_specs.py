class Paths:
  __conf = {
    # Insert paths/to/local/datasets here
    "sintel_mpi": "/path/to/Sintel"
  }

  __splits = {
    # Used for dataloading internally
    "sintel_train": "training",
    "sintel_eval": "test"
  }

  @staticmethod
  def config(name):
    return Paths.__conf[name]

  @staticmethod
  def splits(name):
    return Paths.__splits[name]

class Conf:
  __conf = {
    # Change the following variables according to your system setup.
    "useCPU": False,  # affects all .to(device) calls
  }

  @staticmethod
  def config(name):
    return Conf.__conf[name]


class ProgBar:
  __settings= {
      "disable": False,
      "format_eval": "{desc:19}:{percentage:3.0f}%|{bar:40}{r_bar}",
      "format_train": "{desc:13}:{percentage:3.0f}%|{bar:40}{r_bar}"
  }

  @staticmethod
  def settings(name):
    return ProgBar.__settings[name]
