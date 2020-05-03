"""
This script convert .mat data file to .csv file.
"""
import pandas as pd
import tables


if __name__ == '__main__':

    # Load data file.
    data = tables.open_file('./smtp.mat')

    # Read X, y variables.
    X = data.root.X[:]
    y = data.root.y[:]
    X = X.transpose()
    y = y.transpose()

    # Create a dataframe from variables.
    df = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
    df['y'] = y

    # Save on the disk as csv file.
    df.to_csv('./smtp.csv', index=False)
