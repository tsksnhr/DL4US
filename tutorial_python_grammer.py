
def test_func():

    a = 123
    b = "xyz"
    c = "123"

    return a, b, c

print(test_func())

_, *ret = test_func()
print(_)
print(ret)
