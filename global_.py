class Global:
    __dict = {}

    def __init__(self):
        pass

    def __getitem__(self, key):
        return Global.__dict[key]

    def __setitem__(self, key, value):
        Global.__dict[key] = value
