from collections import OrderedDict

od1 = OrderedDict(a=1, b=2)
od2 = OrderedDict(a=1, b=2)
print(od1, od2)

print(od1.keys() == od2.keys())
print(od1.values() == od2.values())
print(od1.items() == od2.items())

d1 = dict(a=1, b=2)
d2 = dict(a=1, b=2)
print(d1, d2)

print(d1.keys() == d2.keys())
print(d1.values()) == d2.values()
print(d1.items() == d2.items())