# 用于辅助cost分配的class


class Cost_helper:

    def __init__(self, x, y, aoi, max_cost, init_cost=0.0):
        self.x = x
        self.y = y
        self.aoi = aoi
        self.max_cost = max_cost
        self.cost = init_cost

