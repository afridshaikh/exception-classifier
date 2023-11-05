import pandas as pd
import random

# Sample exception messages
runtime_exceptions = [
    "java.lang.NullPointerException: Null value found",
    "java.lang.IllegalArgumentException: Invalid argument",
    "java.lang.IndexOutOfBoundsException: Index out of bounds",
    "java.lang.UnsupportedOperationException: Operation not supported"
]

business_exceptions = [
    "com.myapp.BusinessException: Invalid input data",
    "com.myapp.BusinessException: Payment failed due to insufficient funds",
    "com.myapp.BusinessException: User not found",
    "com.myapp.BusinessException: Product out of stock"
]

data = {
    'Exception': [],
    'Category': []
}

# Generate a dataset with 10,000 samples
for _ in range(10000):
    if random.random() < 0.5:
        data['Exception'].append(random.choice(runtime_exceptions))
        data['Category'].append('runtime')
    else:
        data['Exception'].append(random.choice(business_exceptions))
        data['Category'].append('business')

df = pd.DataFrame(data)

# Save the dataset to a CSV file for training
df.to_csv('exception_training_data_large.csv', index=False)

