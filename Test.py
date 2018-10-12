from functools import reduce

import requests

s = [1, 2, 3, 4, 5, 6, ]
print(s[-3:-1])
print(s[-3::-1])
print(s[-3:-5:-1])
print(s[-3::1])
print(s[1:3])
print(s[1::3])

s1 = s[:]
print(s1)

s = s[1:4]
print(s1)
print(s)

mask = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]


def norm(number):
    return map(lambda m: 0.9 if number & m else 0.1, mask)


print(11 / 2)
print(11 // 2)
