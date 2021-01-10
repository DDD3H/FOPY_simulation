f = open('FOPYparain_INPUT.txt', 'r', encoding='UTF-8')
convert_PMT = f.readlines()
PMT = [float(i.replace('\n', '')) for i in convert_PMT]
# print(PMT, type(PMT[1]))
