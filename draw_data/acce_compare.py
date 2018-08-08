from pandas import read_csv
import matplotlib.pyplot as plt
data = read_csv("C:/Users/binbin/Desktop/acce_compare.csv")
#print(data)
dataset = data.values
print(dataset)
plt.plot(dataset)
plt.xlabel("Number of samples")
plt.ylabel("Acceleration(m/s²)")
plt.legend(['Xiaomi6', 'Galaxy Note3'],loc = 'upper right')
plt.savefig("acce_compare.png")
plt.show()