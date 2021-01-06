linhas = ['linha 1 meu brodi\n', 'linha 2 meu brodi\n', 'linha 3 meu brodi\n', 'linha 4 meu brodi\n']

with open('example', 'w',) as File1:
    for line in linhas:
        File1.write(line)

