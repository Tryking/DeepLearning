"""
Python Numpy Tutorial
"""


# quicksort
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


print(quicksort([3, 6, 8, 10, 1, 2, 1]))

# basic data types
hello = 'hello'  # String literals can use single quotes
world = "world"  # or double quotes; it does not matter.
print(hello)  # Prints "hello"
print(len(hello))  # String length; prints "5"
hw = hello + ' ' + world  # String concatenation
print(hw)  # prints "hello world"
hw12 = '%s %s %d' % (hello, world, 12)  # sprintf style string formatting
print(hw12)  # prints "hello world 12"

# Loops
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))

"""
A tuple is an (immutable) ordered list of values. 
A tuple is in many ways similar to a list; 
one of the most important differences is that tuples can be used as keys in dictionaries and as elements of sets, 
while lists cannot. Here is a trivial example:
"""
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)  # Create a tuple
print(type(t))  # Prints "<class 'tuple'>"
print(d)
print(d[t])  # Prints "5"
print(d[(1, 2)])  # Prints "1"
