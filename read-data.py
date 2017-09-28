filename = "train.csv"

houses = []
with open(filename) as infile:
    lines = infile.readlines()
    headers = lines[0].strip("\n").split(",")

    for j in range(1, len(lines)):
        dict = {}
        values = lines[j].strip("\n").split(",")
        for i in range(len(values)):
            dict[headers[i]] = values[i]
        houses.append(dict)

indata = {"houses" : houses}