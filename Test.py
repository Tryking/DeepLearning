from functools import reduce

import requests

upstream = []
output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, upstream, 0)
print(output)

s = lambda ret, conn: ret * conn
print(s(2, 3))
