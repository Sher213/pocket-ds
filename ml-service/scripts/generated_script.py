
# Execute the generated command
import pandas as pd
import numpy as np

#Load the csv file.
df = pd.read_csv('datasets/20250314_144422_insurance.csv')

#Replace the zero values on the 'children' column with a very small value to avoid dividing by zero error later on.
df['children'] = df['children'].replace(0, np.nan)

#Calculate the charges per children by dividing 'charges' column by 'children' column.
df['charges_per_children'] = df['charges'] / df['children']

#Replace the nan values back to zero in both 'children' and 'charges_per_children' columns.
df['children'] = df['children'].replace(np.nan, 0)
df['charges_per_children'] = df['charges_per_children'].replace(np.nan, 0)

df.head()

# Save the modified dataset
df.to_csv('datasets/20250314_144422_insurance.csv', index=False)
print('Dataset updated successfully.')
