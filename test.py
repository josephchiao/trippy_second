from matplotlib import pyplot as plt
import numpy as np
import math

credits = np.array([[4,4,4,3, 0, 0], [4,3,3,1,3,3], [3,3,4,3,2, 0], [3,3,4,3,1, 0], [3,3,3,3,3, 0], [3,3,3,3,3, 0]])
number_of_course = np.array([4, 6, 5, 5, 5, 5])
marks = np.array([[83, 83, 80, 91, 0, 0], [70, 76, 91, 81, 70, 69], [71, 68, 85, 86, 85, 0], [87, 66, 80, 72, 79, 0], [76, 61, 66, 78, 86, 0], [77, 63, 45, 56, 78, 0]])
averages = np.array([[80, 77, 74, 82, 0, 0], [80, 69, 79, 86, 82, 76], [79, 79, 75, 82, 90, 0], [83, 90, 73, 71, 85, 0], [72, 73, 70, 78, 84, 0], [77, 80, 68, 75, 74, 0]])
delta = (marks - averages) * credits

mean_delta = np.sum(delta, axis=1) / np.sum(credits, axis=1)
plt.plot(mean_delta)
print(mean_delta)
plt.title('Credit corrected average delta (Per credit)')
plt.show()

mean_delta = np.sum(marks - averages, axis=1) / number_of_course
plt.plot(mean_delta)
print(mean_delta)
plt.title("Credit corrected average delta (Per course)")
plt.show()