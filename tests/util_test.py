# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

def make_two_clusters():
    schema = {
        'R1': ('D1', 'D2')
    }
    items_D1 = [
        list(range(0, 10)) + list(range(20,30)),
        list(range(10, 20)),
    ]
    items_D2 = [
        list(range(0, 20)),
        list(range(20, 40)),
    ]
    data_d10_d20 = [((i, j), 0) for i in items_D1[0] for j in items_D2[0]]
    data_d10_d21 = [((i, j), 1) for i in items_D1[0] for j in items_D2[1]]
    data_d11_d20 = [((i, j), 1) for i in items_D1[1] for j in items_D2[0]]
    data_d11_d21 = [((i, j), 0) for i in items_D1[1] for j in items_D2[1]]
    data = data_d10_d20 + data_d10_d21 + data_d11_d20 + data_d11_d21
    return schema, items_D1, items_D2, data
