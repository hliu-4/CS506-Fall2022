import csv

def read_csv(csv_file_path):
    """
        Given a path to a csv file, return a matrix (list of lists)
        in row major.
    """
    list = []
    row = -1
    
    with open(csv_file_path, 'r') as file:
        lines = csv.reader(file)
        for line in lines:
            list.append([])
            row += 1
            for value in line:
                try:
                    list[row].append(int(value))
                except:
                    list[row].append(value)

    return list
