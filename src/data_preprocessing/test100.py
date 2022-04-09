import json

ids = []
for line in open("data/test100/test100.tabular"):
    line_data = line.split("\t")
    if line_data[0].split("|")[3] not in ids:
        ids.append(line_data[0].split("|")[3])


id_clade_dict = dict.fromkeys(ids)

for id in id_clade_dict.keys():
    clades = []
    for new in open("data/test100/test100.tabular"):
        data = new.split("\t")
        if data[0].split("|")[3] == id:
            clades.append(data[1])
    id_clade_dict[id] = clades

print(json.dumps(id_clade_dict, indent=4))
