import numpy as np
import matplotlib.pyplot as plt

x= np.linspace(5,1,200)
y= np.linspace(-5,5,100)
X, Y= np.meshgrid(x,y)

Z= np.sin(np.sqrt(-X**2 + Y**-2))
plt.contour(X,Y,Z, levels=50, cmap='plasma')
plt.colorbar(label='Heigh')
plt.title('Contour Plot')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.show()

