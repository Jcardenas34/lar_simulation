#! /usr/bin/python

from multiprocessing import Pool

def f(x):
    return x*x

def main(args):
    p = Pool(5)
    print(p.map(f, [1, 2, 3]))

if __name__ == '__main__':

    args=1
    main(args)