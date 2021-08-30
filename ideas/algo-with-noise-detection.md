# Algorytm z detekcją szumów
1. Bierzemy dane osób zdrowych (oraz być może osób chorych)
2. Tworzymy następujący pipeline: 
    - model odpowiedzialny za redukcję szumów (jako szum traktujemy również zmiany patalogiczne)

    - model, który dostaje na wejściu **obraz oryginalny** i obraz po **redukcji szumów** i musi stwierdzić, czy te obrazy są podobne do siebie. 
    Trenujemy ten model w taki sposób, żeby w przypadku jeśli przez niego przechodzi obraz płuc zdrowych on twierdził, że 
    obrazy są podobne do siebie, a przypadku jeśli to jest model płuc chorych twierdził, że są różne. 
    (innymi słowy, jeśli zaszumienie na obrazie nie jest duże (płuca zdrowe), model powinien stwiedzić, że te obrazy są do siebie podobne, natomiast jeśli zaszumienie jest wielkie, to model powinien wskazać, że obrazy są różne -> płuca są chore)
    

    ----- Do tego etapu przechodzą tylko płuca uznane za chore ------
    - model który ma za zadanie wykonać segmentację obrazu (przypisać labelki do poszczególnych obiektów). On nie będzie wskazywać
    gdzie dokładnie jest nowotwór, ale on będzie w stanie pokazać poszczególne obiekty, z których specjalista już będzie w stanie wybrać ten 
    który jego interesuje.



# Arch
1. Redukcja szumów (po to żeby użyć sieć syjamską w kroku 2)
2. Klasyfikacja zdrowe vs chore 
3. Chore do sieci Bayes.

Główną zaletą jest to, że dzięki klasyfikacji która usuwa płuca zdrowe zwiększamy poziom zaufania do niepewności sieci Bayes.




