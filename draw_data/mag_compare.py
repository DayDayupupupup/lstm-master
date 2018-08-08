from pandas import read_csv
import matplotlib.pyplot as plt
data = read_csv("mag_compare.csv")
#print(data)
dataset = data.values
print(dataset)
plt.plot(dataset)
plt.xlabel("Number of samples")
plt.ylabel("Magnetic field(uT)")
plt.legend(['Xiaomi6', 'Galaxy Note3'],loc = 'upper right')
plt.savefig("mag_compare.png")
plt.show()