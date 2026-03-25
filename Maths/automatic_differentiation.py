# Forward Pass Automatic Differentiation Algorithm
import math

class Dual:
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual
        # real + dual epsilon

    def __add__(self, other):
        if(isinstance(other, Dual)):
            real = self.real + other.real
            dual = self.dual + other.dual
            return Dual(real, dual)
        return Dual(self.real + other, self.dual)
    __radd__ = __add__
        
    def __mul__(self, other):
        if(isinstance(other, Dual)):
            real = (self.real * other.real)
            dual = (self.real * other.dual) + (self.dual * other.real)
            return Dual(real, dual)
        return Dual(self.real * other, self.dual * other)
    __rmul__ = __mul__

    def sin(self):
        return Dual(
            math.sin(self.real),
            self.dual * math.cos(self.real)
        )

    def exp(self):
        val = math.exp(self.real)
        return Dual(val, self.dual * val)
        
def diff(f, x):
    return f(Dual(x, 1)).dual
