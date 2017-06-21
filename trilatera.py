'''
practicar trilateracion 

'''

# %%
import numpy as np
import numpy.linalg as ln
import matplotlib.pyplot as plt
import numdifftools as ndf

# %%
kml_file = "/home/sebalander/Code/VisionUNQextra/trilateracion/trilat.kml"

# %%
texto = open(kml_file, 'r').read()

names = list()
data = list()

for line in texto.splitlines():
    line = line.replace("\t","")
    
    if line.find("name") is not -1:
        name = line[6:8]
        #print(name)
    
    if line.find("coordinates") is not -1:
        coords = line[13:-14]
        lon, lat, _ = coords.split(sep=',')
        names.append(name)
        data.append([float(lon), float(lat)])
    
        #print(data[-1])

data = np.array(data).T
data, names

xGPS = data.T


# plot data gathered from xml file
fig, ax = plt.subplots()
ax.scatter(data[0], data[1])
ax.set_xlim([min(data[0]) - 2e-5, max(data[0]) - 2e-5])
ax.set_ylim([min(data[1]) - 2e-5, max(data[1]) - 2e-5])
for i, tx in enumerate(names):
    ax.annotate(tx, (data[0,i], data[1,i]))


# %% distancias medidas con bosch glm 250 vf
# cargar las distancias puestas a mano en Db (bosch)
# tambien las distancias sacada de google earth Dg

Db = np.zeros((8, 8, 3), dtype=float)

Db[0, 1] = [8.6237, 8.6243, 8.6206]
Db[0, 2] = [7.0895, 7.0952, 7.0842]
Db[0, 3] = [13.097, 13.104, 13.107]
Db[0, 4] = [18.644, 18.642, 18.641]
Db[0, 5] = [24.630, 24.649, 24.670]
Db[0, 6] = [25.223, 25.218, 25.219]
Db[0, 7] = [41.425, 41.391, 41.401]

Db[1, 2] = [3.8999, 3.8961, 3.89755]  # la tercera la invente como el promedio
Db[1, 3] = [14.584, 14.619, 14.6015]  # tercera inventada
Db[1, 4] = [17.723, 17.745, 17.775]
Db[1, 5] = [25.771, 25.752, 25.752]
Db[1, 6] = [22.799, 22.791, 22.793]
Db[1, 7] = [41.820, 41.827, 41.826]

Db[2, 3] = [10.678, 10.687, 10.682]
Db[2, 4] = [14.287, 14.262, 14.278]
Db[2, 5] = [22.016, 22.003, 22.002]
Db[2, 6] = [19.961, 19.962, 19.964]
Db[2, 7] = [38.281, 38.289, 38.282]  # * medicion corrida 37cm por un obstaculo

Db[3, 4] = [6.3853, 6.3895, 6.3888]
Db[3, 5] = [11.640, 11.645, 11.644]
Db[3, 6] = [13.599, 13.596, 13.606]
Db[3, 7] = [28.374, 28.371, 28.366]

Db[4, 5] = [8.8504, 8.8416, 8.8448]
Db[4, 6] = [7.2463, 7.2536, 7.2526]
Db[4, 7] = [24.088, 24.086, 24.087]

Db[5, 6] = [10.523, 10.521, 10.522]
Db[5, 7] = [16.798, 16.801, 16.797]

Db[6, 7] = [20.794, 20.786, 20.788]

Db -= 0.055  # le resto 5.5cm porque medimos desde la bas en lugar de l centro
indAux = np.arange(len(Db))
Db[indAux, indAux] = 0.0


dg = np.zeros(Db.shape[:2], dtype=float)

dg[0, 1] = 8.27
dg[0, 2] = 6.85
dg[0, 3] = 13.01
dg[0, 4] = 18.5
dg[0, 5] = 24.79
dg[0, 6] = 25.07
dg[0, 7] = 41.22

dg[1, 2] = 3.65
dg[1, 3] = 14.53
dg[1, 4] = 17.63
dg[1, 5] = 26.00
dg[1, 6] = 22.82
dg[1, 7] = 41.65

dg[2, 3] = 10.89
dg[2, 4] = 14.53
dg[2, 5] = 22.55
dg[2, 6] = 20.22
dg[2, 7] = 38.43

dg[3, 4] = 6.37
dg[3, 5] = 11.97
dg[3, 6] = 13.30
dg[3, 7] = 28.26

dg[4, 5] = 9.17
dg[4, 6] = 6.93
dg[4, 7] = 24.05

dg[5, 6] = 10.40
dg[5, 7] = 16.41

dg[6, 7] = 20.49


# las hago simetricas para olvidarme el tema de los indices
triuInd = np.triu_indices(8)
dg.T[triuInd] = dg[triuInd]
Db.transpose([1,0,2])[triuInd] = Db[triuInd]


db = np.mean(Db, axis=2)

plt.figure()
plt.imshow(np.hstack([db, dg]))

plt.figure()
plt.imshow(db - dg)

plt.figure()
plt.imshow((db - dg) / db)

# %% hacemos trilateracion sencilla para sacar condiciones iniciales
# el array con las coordenadas de todos los puntos


def trilateracion(d, signos=None):
    '''
    calcula las posiciones de todos los puntos a partir de los dos primeros
    tomados como origen y direccion de versor x
    si se provee una lista de signos, se corrige la direccion y
    '''
    d2 = d**2  # para usar las distancias al cuadrado
    X = np.empty((d.shape[0], 2), dtype=float)
    # ptos de base
    X[0, 0] = 0  # origen de coords
    X[0, 1] = 0
    X[1, 0] = d[0, 1]  # sobre el versor x
    X[1, 1] = 0
    
    X[2:, 0] = (d2[0, 1] + d2[0, 2:] - d2[1, 2:]) / 2 / d[0, 1]
    X[2:, 1] = np.sqrt(d2[0, 2:] - X[2:, 0]**2)
    
    if signos is not None:
        X[2:, 1] *= signos
    
    return X


Xb = trilateracion(db)
Xg = trilateracion(dg)

fig, ax = plt.subplots()
ax.scatter(Xb.T[0], Xb.T[1], label='bosch')
ax.scatter(Xg.T[0], Xg.T[1], label='google-earth')
for i, tx in enumerate(names):
    ax.annotate(tx, (Xb[i, 0], Xb[i, 1]))
ax.legend()

# %% función error

def x2flat(x):
    '''
    retorna los valores optimizables de x como vector
    '''
    return np.concatenate([[x[1,0]], np.reshape(x[2:], -1)])

def flat2x(xF):
    '''
    reteorna las coordenadas a partir del vector flat
    '''
    return np.concatenate([ np.array([[0.0, 0],[xF[0], 0.0]]),
                    np.reshape(xF[1:], (-1,2))])

def distEr(d1, d2):
    '''
    metrica del error del vector de distancias
    '''
    return np.sqrt(np.mean((d1 - d2)**2))

def dists(xF):
    '''
    calcula la matriz de distancias , solo la mitad triangular superior
    '''
    x = flat2x(xF)
    n = x.shape[0]
    d = np.zeros((n, n), dtype=float)
    
    for i in range(n-1):  # recorro cada fila
        d[i, i+1:] = ln.norm(x[i+1:] - x[i], axis=1)
    
    return d[np.triu_indices(n, k=1)]


# los indices para leer las distancias de la matriz
upInd = np.triu_indices_from(db, k=1)
# defino el jacobiano de las distancias vs x
Jd = ndf.Jacobian(dists)

def newtonOpt(Xb, db, ep=1e-15):
    errores = list()
    d = db[upInd]
    
    #print("cond inicial")
    xF = x2flat(Xb)
    D = dists(xF)
    errores.append(distEr(d, D))
    #print(errores[-1])

    # hago un paso
    j = Jd(xF)
    xFp = xF + ln.pinv(j).dot(d - D)
    D = dists(xFp)
    errores.append(distEr(d, D))
    xF = xFp
    
    # mientras las correcciones sean mayores aun umbral
    while np.mean(np.abs(xFp - xF)) > ep:
    # for i in range(10):
        #print("paso")
        j = Jd(xF)
        xFp = xFp + ln.pinv(j).dot(d - D)
        D = dists(xF)
        errores.append(distEr(d, D))
        xF = xFp
        #print(errores[-1])
    
    return flat2x(xFp), errores

xBOpt, e1 = newtonOpt(Xb, db)
xGOpt, e2 = newtonOpt(Xg, dg)

# %%
fig, ax = plt.subplots()
ax.scatter(xBOpt.T[0], xBOpt.T[1], label='bosch optimo')
ax.scatter(Xb.T[0], Xb.T[1], label='bosch inicial')
for i, tx in enumerate(names):
    ax.annotate(tx, (xBOpt[i, 0], xBOpt[i, 1]))
ax.legend()


fig, ax = plt.subplots()
ax.scatter(xGOpt.T[0], xGOpt.T[1], label='google earth optimo')
ax.scatter(Xg.T[0], Xg.T[1], label='google earth inicial')
for i, tx in enumerate(names):
    ax.annotate(tx, (xGOpt[i, 0], xGOpt[i, 1]))
ax.legend()



fig, ax = plt.subplots()
ax.scatter(xBOpt.T[0], xBOpt.T[1], label='bosch optimo')
ax.scatter(xGOpt.T[0], xGOpt.T[1], label='google-earth optimo')
for i, tx in enumerate(names):
    ax.annotate(tx, (xBOpt[i, 0], xBOpt[i, 1]))
ax.legend()


# %% calculo la rototraslacion y escaleo mas apropiados para pasar de uno a otro

def findrototras(x1, x2):
    '''
    recibe lista de coordenadas x1, x2 (-1, 2)
    y retorna la matriz que transforma x2 = T*x1
    con T de 2x3
    '''
    x, y = x1.T
    ons = np.ones_like(x)
    zer = np.zeros_like(x)
    Ax = np.array([x, y, ons, zer])
    Ay = np.array([y, -x, zer, ons])
    A = np.hstack((Ax, Ay)).T
    
    print(Ax.shape, A.shape)
    
    Y = np.hstack((x2[:, 0], x2[:, 1]))
    
    print(Y.shape)
    
    sol = ln.pinv(A).dot(Y)
    
    R = np.array([[sol[0], sol[1]],
                  [-sol[1], sol[0]]])
    
    T = np.array([[sol[2]], [sol[3]]])
    
    return R, T


# %%
gps2b = findrototras(xGPS, xBOpt)
xBFromGPS = (np.dot(gps2b[0], xGPS.T) + gps2b[1]).T


# %%
fig, ax = plt.subplots()

for i, tx in enumerate(names):
    ax.annotate(tx, (xBOpt[i, 0], xBOpt[i, 1]))

ax.scatter(xBOpt.T[0], xBOpt.T[1], label='distancias bosch')
ax.scatter(xBFromGPS.T[0], xBFromGPS.T[1], label='from coord GPS')
#ax.scatter(xGOpt.T[0], xGOpt.T[1], label='distancias google earth')

ax.legend()




# %%

dbFlat = db[upInd]
dgFlat = dg[upInd]

dif = dgFlat - dbFlat



plt.figure()
plt.scatter(dbFlat, dgFlat - dbFlat)


np.cov(dif)
