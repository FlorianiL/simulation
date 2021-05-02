import random
from matplotlib import pyplot as plt

from chi_squarred import chi_squared_uniform, chi_squared_continuous
from gap import gap_test_discrete, gap_test_continue
from kolmogorov_smirnov import kolmogorov_smirnov
from poker import poker_test
from generators import *


def open_file():
    with open("exp.txt", "r") as e:
        for line in e:
            line = line.strip()
            if "." in line:
                line = line.split(".")[1]
            for c in line:
                yield int(c)


if __name__ == '__main__':

    ## Répartition des décimales d’exponentielle
    ### Premier aperçu
    e_numbers = np.array(list(open_file()))
    print(f"Les 2.000.000 premières décimales :\n"
          f"{e_numbers}")
    e_labels, e_counts = np.unique(e_numbers, return_counts=True)
    print(f"Les chiffres apparaissant dans les décimales :\n"
          f"{e_labels}")
    print(f"Leur fréquences d\'apparition : \n"
          f"{e_counts}")
    df_e_numbers = {"decimals": e_numbers}
    plt.figure()
    plt.bar(e_labels, e_counts, color='palegreen')
    plt.savefig('histo_exp.png')
    plt.show()

    ### Test du Chi Carré
    print(f"Test du Chi-Carré pour les décimales : \n"
          f"{chi_squared_uniform(e_counts)}")

    ### Test du Poker
    print(f"Test du Poker pour les décimales : \n"
          f"{poker_test(e_numbers)}")

    ### Test du gap
    print(f"Test du Gap pour les décimales : \n "
          f"{gap_test_discrete(e_numbers, 0, 5)}")

    ## Générateurs de nombres aléatoires
    ### Techniques employées
    rng1 = Generator1(50)
    rng2 = Generator2(50)
    rng3 = Generator3(50)
    gen_numbers_1 = [rng1.random() for _ in range(100)]
    gen_numbers_2 = [rng2.random() for _ in range(100)]
    gen_numbers_3 = [rng3.random() for _ in range(100)]
    print(f"Résultats des générateurs : \n"
          f"Générateur 1 : {gen_numbers_1} \n"
          f"Générateur 2 : {gen_numbers_2} \n"
          f"Générateur 3 : {gen_numbers_3} \n")

    ### Test du Chi Carré
    print(f"Test du Chi Carré de nos générateurs : \n"
          f"1 --> {chi_squared_continuous(gen_numbers_1)} \n"
          f"2 --> {chi_squared_continuous(gen_numbers_2)} \n"
          f"3 --> {chi_squared_continuous(gen_numbers_3)}")

    ### Test de Kolmogorov-Smirnov
    print(f"Test de Kolmogorov-Smirnov pour notre générateur : \n"
          f"1 --> {kolmogorov_smirnov(gen_numbers_1)} \n"
          f"2 --> {kolmogorov_smirnov(gen_numbers_2)} \n"
          f"3 --> {kolmogorov_smirnov(gen_numbers_3)}")

    ### Test du gap
    print(f"Test du gap pour notre générateur : \n"
          f"1 --> {gap_test_continue(gen_numbers_1, 0.0, 0.5)} \n"
          f"2 --> {gap_test_continue(gen_numbers_2, 0.0, 0.5)} \n"
          f"3 --> {gap_test_continue(gen_numbers_3, 0.0, 0.5)}")

    ### Comparaison avec le générateur de Python
    python_numbers = []
    quantity = len(gen_numbers_1)
    for _ in range(quantity):
        python_numbers.append(random.uniform(0, 1))

    plt.figure()
    plt.hist(gen_numbers_1, color='palegreen', histtype='barstacked')
    plt.hist(python_numbers, color='darkblue', histtype='step')
    plt.legend({'Premier générateur', 'Python'}, loc=4)
    plt.savefig('generator1.png')
    plt.show()

    plt.figure()
    plt.hist(gen_numbers_2, color='palegreen', histtype='barstacked')
    plt.hist(python_numbers, color='darkblue', histtype='step')
    plt.legend({'Second générateur', 'Python'}, loc=4)
    plt.savefig('generator2.png')
    plt.show()

    plt.figure()
    plt.hist(gen_numbers_3, color='palegreen', histtype='barstacked')
    plt.hist(python_numbers, color='darkblue', histtype='step')
    plt.legend({'Troisième générateur', 'Python'}, loc=4)
    plt.savefig('generator3.png')
    plt.show()
