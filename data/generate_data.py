import pandas as pd
import random
from runtime_exceptions import runtime_exceptions
from business_exceptions import business_exceptions

business_exceptions = [
    "InvalidInputException: Invalid user input",
    "DuplicateRecordException: Record already exists in the database",
    "InsufficientFundsException: Account balance is too low",
    "InvalidPermissionException: Insufficient permissions for the operation",
    "InvalidRequestException: Request parameters are invalid",
    "ExpiredSessionException: Session has expired",
    "DataNotFoundException: Required data not found",
    "AuthenticationException: Failed authentication",
    "ConfigurationErrorException: Error in system configuration",
    "TransactionFailedException: Failure in processing transaction"
]

data = []
for _ in range(100000): 
    data.append({'Exception': random.choice(runtime_exceptions), 'Category': 'runtime_exception'})
    data.append({'Exception': random.choice(business_exceptions), 'Category': 'business_exception'})

df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('data/exception_training_data_large.csv', index=False)
