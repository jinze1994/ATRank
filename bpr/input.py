class DataInput:
  def __init__(self, data, batch_size):
    self.batch_size = batch_size
    self.data = data
    self.epoch_size = self.data.shape[0] // self.batch_size
    if self.epoch_size * self.batch_size < self.data.shape[0]:
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):
    if self.i == self.epoch_size:
      raise StopIteration

    t = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                 self.data.shape[0])]
    self.i += 1

    return self.i, t
