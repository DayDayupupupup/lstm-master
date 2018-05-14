# coding=UTF-8
import csv

out = open('train.csv', 'w', newline='')
csv_writer = csv.writer(out)

with open('sqrtData.csv','r') as csvfile:
    content = []
    data = csv.DictReader(csvfile)
    for row in data:
        #content.append(row['fingerMag'])
        out.write("%0.2f" % float(row['fingerMag'])+'\n')

csv_writer.writerow(content)