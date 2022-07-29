import csv

name = input('Name: ')
year = input('Year: ')
Journal = input('Journal: ')
Innovation = input('Innovation: ')
data = [name, year, Journal, Innovation]
file_name = 'article_data.csv'
with open(file_name, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(data)
