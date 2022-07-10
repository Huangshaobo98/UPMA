import numpy as np

class Persistent:
    def print_slot_verbose(self, eposide, slot, prev_real_aoi, next_real_aoi, prev_obv_aoi, next_obv_aoi, pos, reward, energy):
        print("episode: {}, slot: {}, prev real aoi: {}, next real aoi: {}, "
              "prev obv aoi: {}, next obv aoi: {}, uav position {}, reward: {},"
              "energy left {}".format(eposide, slot, np.sum(prev_real_aoi),
                                      np.sum(next_real_aoi), np.sum(prev_obv_aoi), np.sum(next_obv_aoi),
                                      pos, reward, energy))

    def print_slot_verbose_1(self, episode, slot, real_aoi, obv_aoi, pos, reward, energy, epsilon):
        print("episode: {}, slot: {}, real aoi: {:.2f}, obv aoi: {:.2f}, uav position {}, reward: {:.2f},"
              " energy left {:.4f}, epsilon: {:.4f}".format(episode, slot, np.sum(real_aoi), np.sum(obv_aoi), pos, reward, energy[0], epsilon))
