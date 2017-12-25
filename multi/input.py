import math
import numpy as np

class DataInput:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = math.ceil(len(self.data) / self.batch_size)
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        batch_data = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
        self.i += 1


        u, i, y, i_sl, q_sl, c_sl = [], [], [], [], [], []
        for t in batch_data:
            u.append(t[0])

            assert(len(t[2]) == len(t[3]) and len(t[3]) == len(t[4]))
            i_sl.append(len(t[2]))

            assert(len(t[5]) == len(t[6]))
            q_sl.append(len(t[5]))

            assert(len(t[7]) == len(t[8]))
            c_sl.append(len(t[7]))

            i.append(t[9])
            y.append(t[10])


        hist_i_id = np.zeros([len(batch_data), max(i_sl)], np.int64)
        hist_i_act = np.zeros([len(batch_data), max(i_sl)], np.int64)
        hist_i_time = np.zeros([len(batch_data), max(i_sl)], np.int64)
        k = 0
        for t in batch_data:
            for l in range(len(t[2])):
                hist_i_id[k][l] = t[2][l]
                hist_i_act[k][l] = t[3][l]
                hist_i_time[k][l] = t[4][l]
            k += 1

        hist_q_id = np.zeros([len(batch_data), max(q_sl)], np.int64)
        hist_q_time = np.zeros([len(batch_data), max(q_sl)], np.int64)
        k = 0
        for t in batch_data:
            for l in range(len(t[5])):
                hist_q_id[k][l] = t[5][l]
                hist_q_time[k][l] = t[6][l]
            k += 1

        hist_c_id = np.zeros([len(batch_data), max(c_sl)], np.int64)
        hist_c_time = np.zeros([len(batch_data), max(c_sl)], np.int64)
        k = 0
        for t in batch_data:
            for l in range(len(t[7])):
                hist_c_id[k][l] = t[7][l]
                hist_c_time[k][l] = t[8][l]
            k += 1


        return self.i, \
                (u, i, y, \
                hist_i_id, hist_i_act, hist_i_time, i_sl, \
                hist_q_id, hist_q_time, q_sl, \
                hist_c_id, hist_c_time, c_sl)

