import os

C_dataset = "/home/name/C++2Python/data/leet_code/C++"
python_dataset = "/home/name/C++2Python/data/leet_code/Python"

C_set = set()
python_set = set()

#Initalizing trackers
C_counter = 0
python_counter = 0
C_total = len(os.listdir(C_dataset))
python_total = len(os.listdir(python_dataset))

for filename in os.listdir(C_dataset):
    C_set.add(filename.split(".")[0])

for filename in os.listdir(python_dataset):
    python_set.add(filename.split(".")[0])

for file in C_set:
    if file not in python_set:
        filename = file + ".cpp"
        filepath = os.path.join(C_dataset, filename)
        os.remove(filepath)
        C_counter += 1
        print("{} deleted".format(filename))

for file in python_set:
    if file not in C_set:
        filename = file + ".py"
        filepath = os.path.join(python_dataset, filename)
        os.remove(filepath)
        python_counter += 1
        print("{} deleted".format(filename))

print("{} deleted out of the {} python files".format(python_counter, python_total))
print("{} deleted out of the {} C++ files".format(C_counter, C_total))
