
class Global:
    __dict = {}

    @staticmethod
    def init(function):
        Global.__dict = function()

    def __getitem__(self, key):
        return Global.__dict[key]

    def __setitem__(self, key, value):
        Global.__dict[key] = value
