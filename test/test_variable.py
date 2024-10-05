class A:
    def __init__(self, v):
        self.a = v

class V:
    def __init__(self):
        self.v = 1
    def update(self):
        self.v += 1

V1 = V()
A1 = A(V1)
print(A1.a.v)
V1.update()
print(A1.a.v)