import pandas as pd
import numpy as np
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

space_cfg = [
    {'name':'lr', 'type':'cat', 'categories':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]},
    {'name':'temp', 'type':'cat', 'categories':[1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200]},
    {'name':'weight_decay', 'type':'cat', 'categories':[0.00001, 0.00002, 0.0003, 0.00005, 0.00007, 0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001]},
]

space = DesignSpace().parse(space_cfg)
opt = HEBO(space)

data = {
    'lr':[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.6, 0.5, 0.5, 0.5, 0.5],
    'temp':[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    'weight_decay':[0.00002, 0.00002, 0.00002, 0.00002, 0.00002, 0.00002, 0.00005, 0.00005, 0.00007, 0.0001, 0.0002]
}
acc = np.array([63.78, 91.82666667, 91.93, 91.49333333, 91.73, 91.35666667, 92.71, 92.45, 93.07333333, 93.31, 93.06666667])

rec = pd.DataFrame(data=data)
opt.observe(rec, acc)
opt.suggest(n_suggestions=4)