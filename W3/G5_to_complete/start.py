
import cv2
import numpy as np
import sol_ChanVeseIpol_GDExp
from matplotlib import pyplot as plt
from dataclasses import dataclass
from PIL import Image

@dataclass
class Parameters:
    hi: float
    hj: float
    dt: float
    mu: float
    nu: float
    iterMax: float
    tol: float

# ======>>>>  input data  <<<<=======
# folder with the images
folderInput = './'


# figure name to process
figure_name = 'perro.jpeg'
#figure_name = "circles.png"
#figure_name = "noisedCircles.tif"
#figure_name = "phantom17.bmp"
figure_name = "phantom18.bmp"
#figure_name = "phantom19.bmp"
#figure_name = "madrid.jpeg"
#figure_name = "bosque.jpeg"
#figure_name = "matricula.jpg"
#figure_name = "matricula2.jpg"
#figure_name = "matricula3.png"
#figure_name = "flores.jpg"
#figure_name = "karim.jpeg"
#figure_name = "mapamundi.png"
#figure_name = "bhole.jpg"
#figure_name = "f1.jpg"
#figure_name = "f12.jpg"
#figure_name = "f13.png"
#figure_name = "brain1.png"
#figure_name = "brain2.jpg"
#figure_name = "arbol1.jpg"
#figure_name = "spiderman.jpg"
#figure_name = "spiderman2.jpg"
#figure_name = "spiderman3.jpg"
#figure_name = "clouds.jpg"
#figure_name = "fresas.jpg"
#figure_name = "perro2.jpg"
#figure_name = "perro3.png"
#figure_name = "cuadro.jpg"
#figure_name = "bcn.jpg"
#figure_name = "bcn2.jpg"
#figure_name = "mike.png"
#figure_name = "cara.png"
#figure_name = "bebida.jpg"
#figure_name = "cascada.jpg"
#figure_name = "girasoles.jpg"
#figure_name = "carretera1.jpg"
#figure_name = "limonero.jpg"
#figure_name = "pablo.jpg"
figure_name_final=folderInput+figure_name

I1_c = cv2.imread(figure_name_final,cv2.IMREAD_UNCHANGED)
if I1_c.ndim == 3: I1 = cv2.cvtColor(I1_c, cv2.COLOR_BGR2GRAY)
else: I1 = I1_c

I=I1.astype('float')

if len(I1.shape)>2:
    I = np.mean(I, axis=2)
    
print(I.shape)
# visualize the image
cv2.imshow('Image',I)
cv2.waitKey(0)
cv2.destroyAllWindows()

min_val = np.min(I.ravel())
max_val = np.max(I.ravel())
I = (I.astype('float') - min_val)
I = I/max_val

# show normalized image
cv2.imshow('Normalized image',I)
cv2.waitKey(0)

# height, width, number of channels in image
height_mask = I.shape[0]
width_mask = I.shape[1]
dimensions_mask = I.shape

ni=height_mask
nj=width_mask

#Lenght and area parameters
#circles.png mu=1, mu=2, mu=10
#noisedCircles.tif mu=0.1
#phantom17 mu=1, mu=2, mu=10
#phantom18 mu=0.2 mu=0.5
#hola carola
mu=0.025
nu=0

#Parameters
lambda1=1
lambda2=1
#lambda1=10^-3 #Hola carola problem
#lambda2=10^-3 #Hola carola problem

epHeaviside=1
eta=0.1
#eta=1

tol=0.01
#dt=(10^-2)/mu
dt=(10^-1)/mu
#iterMax=100000
iterMax=300
#reIni=0 #%Try both of them
#reIni=500
reIni=100

X, Y = np.meshgrid(np.arange(0,nj), np.arange(0,ni),indexing='xy')

#Initial phi
# ESTO LO HEMOS CAMBIADO, ANTES ESTABA ni con X y nj con Y
#phi_0=(-np.sqrt((X - nj*2//1)**2 + (Y - ni//2)**2) + 50) + (-np.sqrt((X - nj//10)**2 + (Y - ni//2)**2) + 50)
phi_0=(-np.sqrt((X - nj//2)**2 + (Y - ni//2)**2) + 50)

#phi_0=np.sin((np.pi/10)*X)*np.sin((np.pi/10)*Y)

# This initialization allows a faster convergence for phantom 18
#phi_0=(-np.sqrt( ( X-round(ni/2))**2 + (Y-round(nj/4))**2)+50)
#Normalization of the initial phi to [-1 1]
phi_0=phi_0-np.min(phi_0)
phi_0=2*phi_0/np.max(phi_0)
phi_0=phi_0-1
'''
phi_0=I #For the Hola carola problem

min_val = np.min(phi_0)
max_val = np.max(phi_0)

phi_0=phi_0-min_val
phi_0=2*phi_0/max_val
phi_0=phi_0-1
'''
#Explicit Gradient Descent
seg, dif_dict, images = sol_ChanVeseIpol_GDExp.G5_sol_ChanVeseIpol_GDExp( I, phi_0, mu, nu, eta, lambda1, lambda2, tol, epHeaviside, dt, iterMax, reIni )

# show output image
contours, hierarchy = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#img = cv2.cvtColor(I1_c, cv2.COLOR_GRAY2BGR )
fin = cv2.drawContours(I1_c, contours, -1, (0,255,0), 2)
cv2.imshow('Seg',fin)
cv2.imwrite("seg_" + figure_name.split(".")[0] + ".png", fin)
cv2.waitKey(0)

plt.plot(dif_dict.keys(), dif_dict.values())
plt.title("Evolution of the difference between phi and phi_old over the iterations")
plt.show()

def make_and_show_gif(images, save_name):
    frames = []
    iter = 0
    for image in images:
        
        text = "n iter: " + str(iter)
        if image.ndim == 3:
            image = cv2.rectangle(image, (image.shape[1] - 110, 0), (image.shape[1], 20), (0,255,0), -1)
            cv2.putText(image, text, (image.shape[1] - 100, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        else: 
            image = cv2.rectangle(image, (image.shape[1] - 110, 0), (image.shape[1], 20), (255,255,255), -1)
            cv2.putText(image, text, (image.shape[1] - 100, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        real_img = Image.fromarray(image)
        frames.append(real_img)
        iter += 1
        
    frame_one = frames[0]
    name = "./gifs/" + save_name + ".gif"
    frame_one.save(name, format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)
    print(f"gif correctly saved as: {save_name}")

    imageObject = Image.open("./"+name)
    
make_and_show_gif(images, figure_name.split(".")[0])
#cv2.imwrite("seg_" + figure_name.split(".")[0] + ".png", (seg * 255).astype("int"))


