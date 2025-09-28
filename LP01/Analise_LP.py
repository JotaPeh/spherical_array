from scipy import special as sp
from scipy.optimize import root_scalar
from scipy import integrate as spi
import numpy as np
from functools import partial

def Z_par_g(Dphip, tgef, delt, L, M, Dthetaa, Dphia, thetap, phip, sweep):
    L = int(L)
    M = int(M)

    # Constantes gerais
    e0 = 8.854e-12          # (F/m)
    u0 = np.pi*4e-7         # (H/m)
    gamma = 0.577216        # Constante de Euler-Mascheroni
    eps = 1e-5              # Limite para o erro numérico

    # Geometria da cavidade
    a = 100e-3              # Raio da esfera de terra (m)
    h = 1.524e-3            # Espessura do substrato dielétrico (m)
    a_bar = a + h/2         # Raio médio da esfera de terra (m)
    
    deltaPhic = delt
    deltaThetac = delt

    
    theta1, theta2 =  np.pi/2 - Dthetaa/2, np.pi/2 + Dthetaa/2       # Ângulos polares físicos do patch (rad)
    phi1, phi2 = np.pi/2 - Dphia/2, np.pi/2 + Dphia/2                # Ângulos azimutais físicos do patch (rad)

    theta1c, theta2c = theta1 - deltaThetac, theta2 + deltaThetac      # Ângulso de elevação da cavidade (rad)
    phi1c, phi2c = phi1 - deltaPhic, phi2 + deltaPhic                  # Ângulo de azimutal 1 da cavidade (rad)

    # Coordenadas dos alimentadores coaxiais (rad)
    thetap = [thetap]  
    phip = [phip]
    df = 1.3e-3             # Diâmetro do pino central dos alimentadores coaxiais (m)
    er = 2.55               # Permissividade relativa do substrato dielétrico
    es = e0 * er            # Permissividade do substrato dielétrico (F/m)

    # Funções associadas de Legendre para |z| < 1
    def legP(z, l, m):
        u = m * np.pi / (phi2c - phi1c)
        if u.is_integer():
            return np.where(np.abs(z) < 1, (-1)**u * (sp.gamma(l+u+1)/sp.gamma(l-u+1)) * np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1+u,(1-z)/2)  / sp.gamma(u+1), 1)
        else:
            return np.where(np.abs(z) < 1, np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u), 1)

    def legQ(z, l, m):
        u = m * np.pi / (phi2c - phi1c)
        if u.is_integer():
            return np.where(np.abs(z) < 1, np.pi * (np.cos((u+l)*np.pi) * np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u) \
        - np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1+z)/2)  / sp.gamma(1-u)) / (2 * np.sin((u+l)*np.pi)), 1)
        else:
            return np.where(np.abs(z) < 1, np.pi * (np.cos(u*np.pi) * np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u) \
        - sp.gamma(l+u+1) * np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1+u,(1-z)/2)  / (sp.gamma(1+u) * sp.gamma(l-u+1))) / (2 * np.sin(u*np.pi)), 1)

    # Derivadas
    def DP(theta, l, m):
        z = np.cos(theta)
        u = m * np.pi / (phi2c - phi1c)
        return l*legP(z, l, m)/np.tan(theta) - (l+u) * legP(z, l-1, m) / np.sin(theta)

    def DQ(theta, l, m):
        z = np.cos(theta)
        u = m * np.pi / (phi2c - phi1c)
        return l*legQ(z, l, m)/np.tan(theta) - (l+u) * legQ(z, l-1, m) / np.sin(theta)

    # Início
    def Equation(l, theta1f, theta2f, m):
        return np.where(np.abs(np.cos(theta1f)) < 1 and np.abs(np.cos(theta2f)) < 1, DP(theta1f,l,m)*DQ(theta2f,l,m)-DQ(theta1f,l,m)*DP(theta2f,l,m) , 1)

    Lambda = np.linspace(-0.1, 32, 64) # Domínio de busca, o professor usou a janela de 0.1
    rootsLambda = []

    def root_find(n):
        roots = []
        Wrapper = partial(Equation, theta1f = theta1c, theta2f = theta2c, m = n)
        k = 0
        for i in range(len(Lambda)-1):
            if Wrapper(Lambda[i]) * Wrapper(Lambda[i+1]) < 0:
                root = root_scalar(Wrapper, bracket=[Lambda[i], Lambda[i+1]], method='bisect')
                roots.append(root.root)
                k += 1
        
        # Remoção de raízes inválidas
        k = 0
        for r in roots:
            if r < n * np.pi / (phi2c - phi1c) - 1 + eps and round(r, 5) != 0:
                k += 1
        rootsLambda.append(roots[k:k+5])
    for n in range(0,5):
        root_find(n)

    # Frequências de ressonância (GHz)
    flm = []
    for r in rootsLambda:
        flm.append([round(p,5) for p in 1e-9*np.sqrt([x*(x + 1) for x in r])/(2*np.pi*a_bar*np.sqrt(er*e0*u0))])
    flm = np.transpose(flm) * 1e9                  # Hz
    rootsLambda = np.transpose(rootsLambda)

    def R(v,l,m):                                  # Função auxiliar para os campos elétricos
        Lamb = rootsLambda[l][m]
        return (DP(theta1c,Lamb,m)*legQ(v,Lamb,m) - DQ(theta1c,Lamb,m)*legP(v,Lamb,m)) * np.sin(theta1c)

    kLM = 2*np.pi*flm[L][M]*np.sqrt(u0*es)

    # Impedância de entrada - Circuito RLC:
    def R2(v, l, m):                               # Quadrado da função auxiliar para os campos elétricos
        return R(v, l, m)**2
        
    def RLC(f, klm, L, M, p1, p2):
        U = M * np.pi / (phi2c - phi1c)
        Ilm = spi.quad(partial(R2, l = L, m = M),np.cos(theta2c),np.cos(theta1c))[0]

        # Auxiliar alpha
        alpha = h * (R(np.cos(thetap[p1-1]), L, M) * np.cos(U * (phip[p1-1]-phi1c)) * np.sinc(U * Dphip / (2*np.pi))) * \
            (R(np.cos(thetap[p2-1]), L, M) * np.cos(U * (phip[p2-1]-phi1c)) * np.sinc(U * Dphip / (2*np.pi))) / ((phi2c - phi1c) * es * a_bar**2 * Ilm)
        if M != 0:
            alpha = 2 * alpha
        
        # Componentes RLC
        RLM = alpha/(2 * np.pi * f * tgef)
        CLM = 1/alpha
        LLM = alpha/(2 * np.pi * flm[L][M])**2
        return 1/(1/RLM+1j*(2 * np.pi * f * CLM - 1/(2 * np.pi * f * LLM)))

    def ZLP(f, p1, p2):
        k = (2 * np.pi * f) * np.sqrt(es * u0) 
        eta = np.sqrt(u0 / es)
        Xp = (eta * k * h / (2 * np.pi)) * (np.log(4 / (k * df)) - gamma)
        return RLC(f, kLM, L, M, p1, p2) + 1j*Xp

    return ZLP(np.linspace(sweep[0], sweep[1], sweep[2]), 1, 1)