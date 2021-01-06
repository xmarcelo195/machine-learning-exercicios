class Circle(object):
    def __init__(self, raio, cor):
        self.raio = raio
        self.cor = cor

    def adc_raio(self, r):
        self.raio = self.raio + r
        return self.raio


class Retangulo(object):
    def __init__(self, altura, largura, cor):
        self.altura = altura
        self.largura = largura
        self.cor = cor


redcircle = Circle(2, 'red')

print(type(redcircle))
