class Customer(object):
    def __init__(self, location, timer):
        self.location = location
        self.transit = False
        self.timer = timer
        
    def __str__(self):
        return "De"