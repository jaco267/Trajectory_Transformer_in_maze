import torch as tc
a = [[ 3,  1, 18],
        [ 0,  1, 38],
        [ 4,  1,  5],
        [ 4,  1, 13],
        [ 4,  1, 15],
        [ 1,  1, 38],
        [ 1,  1, 25],
        [ 0,  1, 11],
        [ 1,  1, 10],
        [ 0,  1, 16],
        [ 4,  1, 12],
        [ 4,  1, 35],
        [ 3,  1, 17],
        [ 0,  1, 30],
        [ 2,  1,  9],
        [ 2,  1, 11],
        [ 2,  1, 18],
        [ 1,  1, 32],
        [ 4,  1, 12],
        [ 4,  1, 26],
        [ 4,  1, 16],
        [ 0,  1, 13],
        [ 0,  1, 27],
        [ 2,  1, 17],
        [ 4,  1, 32],
        [ 2,  1,  4],
        [ 0,  1, 30],
        [ 4,  1, 13],
        [ 1,  1,  1],
        [ 1,  1, 33],
        [ 1,  1, 12],
        [ 4,  1,  5],
        [ 4,  1,  3],
        [ 0,  1,  1],
        [ 0,  1, 36],
        [ 1,  1, 35],
        [ 2,  1,  1],
        [ 0,  1, 22],
        [ 0,  1, 14],
        [ 2,  1, 27],
        [ 1,  1, 25],
        [ 1,  1, 12],
        [ 3,  1, 13],
        [ 3,  1,  8],
        [ 4,  1, 20],
        [ 0,  1, 34],
        [ 1,  1, 30],
        [ 0,  1,  5],
        [ 0,  1, 31],
        [ 0,  1,  7],
        [ 4,  1,  2],
        [ 2,  1, 30],
        [ 2,  1, 39],
        [ 1,  1,  9],
        [ 3,  1, 37],
        [ 1,  1, 22],
        [ 1,  1, 20],
        [ 3,  1, 37],
        [ 1,  1, 11],
        [ 0,  1, 17],
        [ 2,  1,  1],
        [ 0,  1, 17],
        [ 3,  1, 11],
        [ 1,  1, 23],
        [ 3,  1,  3],
        [ 1,  1, 13],
        [ 0,  1, 21],
        [ 0,  1, 31],
        [ 3,  1, 12],
        [ 3,  1, 25],
        [ 2,  1,  5],
        [ 1,  1, 14],
        [ 2,  1, 18],
        [ 0,  1, 21],
        [ 1,  1, 22],
        [ 4,  1, 20],
        [ 4,  1, 22],
        [ 1,  1, 33],
        [ 4,  1, 33],
        [ 2,  1, 38],
        [ 1,  1, 26],
        [ 1,  1, 26],
        [ 2,  1, 25],
        [ 0,  1, 36],
        [ 2,  1, 35],
        [ 0,  1,  2],
        [ 4,  1,  2],
        [ 0,  1, 38],
        [ 4,  1, 14],
        [ 4,  1, 21],
        [ 0,  1, 15],
        [ 0,  1, 26],
        [ 1,  1, 36],
        [ 4,  1, 19],
        [ 0,  1, 37],
        [ 1,  1, 36],
        [ 2,  1, 19],
        [ 0,  1, 19],
        [ 4,  1, 38],
        [ 3,  1, 34],
        [ 2,  1,  1],
        [ 4,  1, 26],
        [ 4,  1, 27],
        [ 0,  1, 10],
        [ 0,  1,  1],
        [ 3,  1, 33],
        [ 1,  1,  1],
        [ 1,  1, 15],
        [ 2,  1, 26],
        [ 0,  1, 16],
        [ 4,  1, 37],
        [ 0,  1, 28],
        [ 3,  1, 14],
        [ 1,  1, 25],
        [ 3,  1, 28],
        [ 1,  1, 27],
        [ 2,  1, 28],
        [ 4,  1, 38],
        [ 0,  1, 37],
        [ 2,  1,  7],
        [ 0,  1, 24],
        [ 3,  1, 17],
        [ 1,  1, 15],
        [ 0,  1, 25],
        [ 2,  1, 18],
        [ 2,  1, 38],
        [ 2,  1, 21],
        [ 2,  1, 37]]



a.sort(key=lambda x: x[2])
print(a)

for ten in a:
    print('action',ten[0],'value',ten[2])