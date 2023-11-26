import pandas as pd
import random
from runtime_exceptions import runtime_exceptions
from business_exceptions import business_exceptions

data = []
for _ in range(100000): 
    data.append({'Exception': random.choice(runtime_exceptions), 'Category': 'runtime_exception'})
    data.append({'Exception': random.choice(business_exceptions), 'Category': 'business_exception'})

df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('data/exception_training_data_large.csv', index=False)
