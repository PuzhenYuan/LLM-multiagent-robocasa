from copy import deepcopy

num1 = 0
def func1():
    global num1
    num1 += 1
    print(num1)

class func2:
    def __init__(self):
        self.num2 = 0
    def __call__(self):
        self.num2 += 1
        print(self.num2)

di = {'func1': func1, 'func2': func2()}

A = deepcopy(di['func1'])
B = deepcopy(di['func1'])

A()
B()

C = deepcopy(di['func2'])
D = deepcopy(di['func2'])

C()
D()

print(A)
print(B)
print(A is B)

print(C)
print(D)
print(C is D)