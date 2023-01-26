from os import listdir
import re

lst = listdir('.')
print(lst)

lst2 = []
for file in listdir('../Data/'):
    if file.endswith('.txt'):
    #if file.endswith('_URSP.txt'):
        lst2.append(file)

m = re.search("([0-9]{2})([0-9]{2})([0-9]{2})\_([0-9]{2})([0-9]{2})([0-9]{2})", lst2[0])
print(m.groups())
m = re.findall("([0-9]{2})", lst2[0])
print(m)

def keys(var):
    return [int(num) for num in re.findall("([0-9]{2})", var)]
    # lambda var: [int(num) for num in re.findall("([0-9]{2})", var)]

lst2.sort(key=keys)
print(lst2)
print(lst[0][0:2])