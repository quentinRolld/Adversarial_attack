# Adversarial_attack partie 5

Step 1 : entraîner un modèle de classification Maxout network avec 240 et 1600 unit sur le dataset MNIST en utilisant une loss normale :
Pour le 240 unit : 
Epoch 68/100.. Train loss: 0.478.. Validation loss: 0.323.. Train accuracy: 86.515.. Validation accuracy: 90.750
Pour le 1600 unit :
Epoch 100/100.. Train loss: 1.014.. Validation loss: 0.696.. Train accuracy: 67.597.. Validation accuracy: 78.910

Moins bon résultats pour le 1600 unit que le 240 unit. Comme je n'ai pas forcément le même réseau, je ne peux pas correctment comparer les résultats avec ceux de l'article.


Step 2: entraîner le même modèle de classification avec loss adversarial:$\tilde{J}(\theta, x, y) = \alpha J(\theta, x, y) + (1 - \alpha) J(\theta, x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y)))$
et comparer les errors rates
param : entraînement sur 100 epochs, batch size 64, learning rate 0.001, $\alpha = 0.5$, $\epsilon = 0.1$.
Pour le 240 unit :
Epoch 100/100.. Train loss: 2.394.. Train accuracy: 86.685.. Validation loss: 2.484.. Validation accuracy: 90.680 //validation accuracy constante tout le long de l'entraînement ( à investiguer)
Pour le 1600 unit :
Epoch 100/100.. Train loss: 1.144.. Train accuracy: 70.547.. Validation loss: 0.907.. Validation accuracy: 82.300

meilleurs perf pour le 1600 unit avec une adversarial loss, max de perf obtenu à Epoch 25/100.. Train loss: 1.312.. Train accuracy: 67.780.. Validation loss: 0.952.. Validation accuracy: 86.940

Step 3 : attaquer ce modèle avec la méthode FGSM et lui donner des images adversariales.

Step 4