# perceptron

def AND(x1, x2):

    w1 = 0.5
    w2 = 0.5
    thr = 0.7

    out = w1*x1+w2*x2

    if out > thr:
        return 1
    else:
        return 0

print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))
