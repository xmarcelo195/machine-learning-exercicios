#txtsimigualnao.txt
#/home/marcelo/PycharmProjects/Data/Parlamentares/


with open('/home/marcelo/PycharmProjects/Data/Parlamentares/txtsimigualnao.txt', 'r', encoding='latin-1') as File1:

    file_stuff = File1.readlines()

    #print(file_stuff)

a = len(file_stuff)
print(a)
i = 0

while i <= (a-1):
    file_stuff[i] = file_stuff[i].rstrip()
    i = i+1
    
print(file_stuff)
