import pickle
from helper import set_DoC

data = []
device = "cpu"

for i in range(0, 10):
    for num in range(0, 5):
        # Including True Class
        for j in range(2, 11):
            v1 = set_DoC(i, 10, j, device, True)
            data.append([i, v1])

        # Excluding True Class
        for k in range(1, 10):
            v2 = set_DoC(i, 10, k, device, False)
            data.append([i, v2])

# Open a file in write-binary mode
with open('doc.pkl', 'wb') as f:
    # Use pickle.dump() t o serialize the data and save it to the file
    pickle.dump(data, f)
