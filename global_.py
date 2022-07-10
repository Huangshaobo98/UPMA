
class Global:
    __dict = {}

    @staticmethod
    def init(_dict):
        Global.__dict = _dict

    def __getitem__(self, key):
        return Global.__dict[key]

    def __setitem__(self, key, value):
        Global.__dict[key] = value
