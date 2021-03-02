import numpy as np
import csv

MichaelJackson = np.array([0.0,0.99,8.64,8.69,12.33,12.96,38.99,30.12,13.99,17.25])
LadyGaga = np.array([0.0,0.99,14.28,11.20,11.25,14.98,13.69,9.99,18.95])
Eminem = np.array([0.0,0.99,15.99,9.33,21.61,22.37,12.80,10.75,11.70])

with open('artist_wiki_page_views-20200101-20201231.csv', newline='') as csvfile:
    freader = csv.reader(csvfile, delimiter=',')
    rcount = 0
    for row in freader:
        print(row)
        if rcount == 0:
            header = row
        else:
            day = row[0]
            for i in range(1,len(row)):
                print(day + "," + header[i] + "," + row[i])
        rcount = rcount + 1