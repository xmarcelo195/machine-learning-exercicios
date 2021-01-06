from random import randint
import collections



resultados = []
i = 1
vezes = 1000000
while i <= vezes:
    lista = []
    j = 1
    while j <= 25:
        lista.append(randint(0, 365))
        j = j+1

    for num in lista:
        repetidos = collections.Counter(lista)
        if repetidos[num] > 1:
            resultados.append(1)
            break
        #print(repetidos[num])
    i = i + 1


print('Porcentagem de = ', (len(resultados)/vezes)*100, '%')


