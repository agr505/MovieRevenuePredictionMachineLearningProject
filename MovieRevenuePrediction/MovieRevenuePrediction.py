import pandas as pd

df = pd.read_csv('/Users/tarushreegandhi/Downloads/the-movies-dataset/movies_metadata.csv')
print(df.columns.tolist())
print('\n')
df = pd.read_csv('/Users/tarushreegandhi/Downloads/the-movies-dataset/credits.csv')
print(df.columns.tolist())
print('\n')
df = pd.read_csv('/Users/tarushreegandhi/Downloads/the-movies-dataset/ratings.csv')
print(df.columns.tolist())
print('\n')


# df = pd.read_csv('/Users/tarushreegandhi/Downloads/the-movies-dataset/movies_metadata.csv', 
#                 usecols=['budget','genres','popularity','release_date','revenue'])

# Crew
# Keywords
# Ratings
# Budget
# Genre
# Popularity
# Production company and country
# Release date
# revenue

# csv2 = csv1.loc[:, ['Acceleration', 'Pressure']]

print(df)


