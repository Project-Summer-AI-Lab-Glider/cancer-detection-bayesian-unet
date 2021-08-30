# Bayesian UNet
1. Bierzemy dane osób zdrowych
2. Dzielimy je na wertykalne plastry (ponieważ są już sieci, 
które były wytrenowane do tego, żeby wykonywać płuc segmentację na takich plastrach https://arxiv.org/abs/1701.08816)
3. Tworzymy sieć bayesowską która będzie dostawać na wejściu **wszystkie plastry jako jeden wektor**??
4. Liczymy na to, że w przypadku jeśli sieć dostanie płuca chorej osoby ona zwróci nam wysoką niepewność na pikselach które odpowiadają nowotworom. 