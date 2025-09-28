from scipy import special as sp
from scipy.optimize import root_scalar
from scipy import integrate as spi
from scipy.interpolate import interp1d
import numpy as np
from functools import partial
import warnings

def synth_ant(flm_des):
    # Constantes gerais
    dtr = np.pi/180         # Graus para radianos (rad/°)
    e0 = 8.854e-12          # (F/m)
    u0 = np.pi*4e-7         # (H/m)
    c = 1/np.sqrt(e0*u0)    # Velocidade da luz no vácuo (m/s)
    gamma = 0.577216        # Constante de Euler-Mascheroni
    eps = 1e-5              # Limite para o erro numérico

    # Geometria da cavidade
    a = 100e-3              # Raio da esfera de terra (m)
    h = 1.524e-3            # Espessura do substrato dielétrico (m)
    a_bar = a + h/2         # Raio médio da esfera de terra (m)
    b = a + h               # Raio exterior do dielétrico (m)

    deltatheta = h/a        # Largura angular do campo de franjas e da abertura polar (rad)
    deltaPhi = h/a          # Largura angular do campo de franjas e da abertura azimutal (rad)

    # Coordenadas dos alimentadores coaxiais (rad)
    df = 1.3e-3             # Diâmetro do pino central dos alimentadores coaxiais (m)
    er = 2.55               # Permissividade relativa do substrato dielétrico
    es = e0 * er            # Permissividade do substrato dielétrico (F/m)

    # Outras variáveis
    tgdel = 0.0022          # Tangente de perdas
    sigma = 5.8e50          # Condutividade elétrica dos condutores (S/m)

    # Funções associadas de Legendre para |z| < 1
    def legP(z, l, u):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = float(u)
            if u.is_integer():
                return np.where(np.abs(z) < 1, (-1)**u * (sp.gamma(l+u+1)/sp.gamma(l-u+1)) * np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1+u,(1-z)/2)  / sp.gamma(u+1), 1)
            else:
                return np.where(np.abs(z) < 1, np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u), 1)

    def legQ(z, l, u):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = float(u)
            if u.is_integer():
                return np.where(np.abs(z) < 1, np.pi * (np.cos((u+l)*np.pi) * np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u) \
            - np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1+z)/2)  / sp.gamma(1-u)) / (2 * np.sin((u+l)*np.pi)), 1)
            else:
                return np.where(np.abs(z) < 1, np.pi * (np.cos(u*np.pi) * np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u) \
            - sp.gamma(l+u+1) * np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1+u,(1-z)/2)  / (sp.gamma(1+u) * sp.gamma(l-u+1))) / (2 * np.sin(u*np.pi)), 1)

    # Derivadas
    def DP(theta, l, u):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = np.cos(theta)
            return l*legP(z, l, u)/np.tan(theta) - (l+u) * legP(z, l-1, u) / np.sin(theta)

    def DQ(theta, l, u):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = np.cos(theta)
            return l*legQ(z, l, u)/np.tan(theta) - (l+u) * legQ(z, l-1, u) / np.sin(theta)

    #########################################################################################################################################################
    # Funções auxiliares de busca de raízes
    def EquationCircTheta(Dtheta, Dphi, m, K):
        theta1, theta2 = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        Lamb = (np.sqrt(1+4*a_bar**2 * K**2)-1)/2
        u = m * np.pi / Dphi
        return np.where(np.abs(np.cos(theta1)) < 1 and np.abs(np.cos(theta2)) < 1, DP(theta1,Lamb,u)*DQ(theta2,Lamb,u)-DQ(theta1,Lamb,u)*DP(theta2,Lamb,u) , 1)

    def EquationCircPhi(Dphi, Dtheta, m, K):
        theta1, theta2 = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        Lamb = (np.sqrt(1+4*a_bar**2 * K**2)-1)/2
        u = m * np.pi / Dphi
        return np.where(np.abs(np.cos(theta1)) < 1 and np.abs(np.cos(theta2)) < 1, DP(theta1,Lamb,u)*DQ(theta2,Lamb,u)-DQ(theta1,Lamb,u)*DP(theta2,Lamb,u) , 1)

    def Theta_find(K, Dphi = 1):
        Thetas = np.linspace(0.1, 359.9, 601) * dtr # Domínio de busca
        roots = []
        Wrapper = partial(EquationCircTheta, Dphi = Dphi,  m = 0, K = K) # Modo TM^r_10 desejado
        k = 0
        for i in range(len(Thetas)-1):
            if Wrapper(Thetas[i]) * Wrapper(Thetas[i+1]) < 0:
                root = root_scalar(Wrapper, bracket=[Thetas[i], Thetas[i+1]], method='bisect')
                roots.append(root.root)
                k += 1

        return np.array(roots)

    def Phi_find(K, Dtheta):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Phis = np.linspace(0.1, 179.9, 351) * dtr # Domínio de busca
            roots = []
            Wrapper = partial(EquationCircPhi, Dtheta = Dtheta,  m = 1, K = K) # Modo TM^r_01 desejado
            k = 0
            for i in range(len(Phis)-1):
                if Wrapper(Phis[i]) * Wrapper(Phis[i+1]) < 0:
                    root = root_scalar(Wrapper, bracket=[Phis[i], Phis[i+1]], method='bisect')
                    roots.append(root.root)
                    k += 1

            return np.array(roots)

    #########################################################################################################################################################
    # Projeto iterativo
    l = m = 35 # m <= l

    Ml, Mm = np.meshgrid(np.arange(0,l+1), np.arange(0,m+1))
    delm = np.ones(m+1)
    delm[0] = 0.5

    def RLC(f, Kex, L, U, thetap, phip, tgef, Dtheta, Dphi):
        Dphip = np.exp(1.5)*df/(2*a_bar*np.sin(thetap))
        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
        Ilm = spi.quad(partial(R2, l = L, m = U, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]

        alpha = h * (R(np.cos(thetap), L, U, theta1c = theta1c) * np.cos(U * (phip-phi1c)) * np.sinc(U * Dphip / (2*np.pi))) * \
            (R(np.cos(thetap), L, U, theta1c = theta1c) * np.cos(U * (phip-phi1c)) * np.sinc(U * Dphip / (2*np.pi))) / ((phi2c - phi1c) * es * a_bar**2 * Ilm)
        if U != 0:
            alpha = 2 * alpha
        
        RLM = alpha/(2 * np.pi * f * tgef)
        CLM = 1/alpha
        LLM = alpha/(2 * np.pi * Kex / (2*np.pi*np.sqrt(u0*es)))**2
        return 1/(1/RLM+1j*(2 * np.pi * f * CLM - 1/(2 * np.pi * f * LLM)))

    def Z(f, k01, k10, thetap, phip, tgef, Dtheta, Dphi):
        L01, M01 = (np.sqrt(1+4*a_bar**2 * k01**2)-1)/2, np.pi/Dphi
        L10, M10 = (np.sqrt(1+4*a_bar**2 * k10**2)-1)/2, 0
        k = (2 * np.pi * f) * np.sqrt(es * u0) 
        eta = np.sqrt(u0 / es)
        Xp = (eta * k * h / (2 * np.pi)) * (np.log(4 / (k * df)) - gamma)
        return RLC(f, k01, L01, M01, thetap, phip, tgef, Dtheta, Dphi) + RLC(f, k10, L10, M10, thetap, phip, tgef, Dtheta, Dphi) + 1j*Xp

    def hankel_spher2(n, x, der = 0):
        return sp.riccati_jn(n, x)[der]/x - 1j * sp.riccati_yn(n, x)[der]/x

    def schelkunoff2(n, x, der = 1):
        return sp.riccati_jn(n, x)[der] - 1j * sp.riccati_yn(n, x)[der]

    def Itheta(theta, L, M):
        if M > L:
            return 0
        return sp.lpmn(M, L, np.cos(theta))[0][M, L] * np.sin(theta)

    def IDtheta(theta, L, M):
        if M > L:
            return 0
        return -sp.lpmn(M, L, np.cos(theta))[1][M, L] * np.sin(theta)**2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S_lm = (2 * Ml * (Ml+1) * sp.gamma(1+Ml+Mm)) / ((2*Ml+1) * sp.gamma(1+Ml-Mm))
        S_lm += (1-np.abs(np.sign(S_lm)))*eps

    def Integrais(theta1, theta2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            I_th = np.array([[spi.quad(partial(Itheta, L = i, M = j), theta1 , theta2)[0] for i in range(l+1)] for j in range(m+1)])
            I_dth = np.array([[spi.quad(partial(IDtheta, L = i, M = j), theta1 , theta2)[0] for i in range(l+1)] for j in range(m+1)])
        return I_th, I_dth

    def R(v,L,m, theta1c):
        return (DP(theta1c,L,m)*legQ(v,L,m) - DQ(theta1c,L,m)*legP(v,L,m)) * np.sin(theta1c)

    def R2(v, l, m, theta1c):
        return R(v, l, m, theta1c)**2

    def tgefTot(klm, mode, Dtheta, Dphi):
        Rs = np.sqrt(klm * np.sqrt(u0/es) / (2 * sigma))
        Qc = klm * np.sqrt(u0/es) * h / (2 * Rs)
        Qc = Qc * (3*a**2 + 3*a*h + h**2) / (3*a**2 + 3*a*h + h**2 * 3/2)
        
        if mode == '10':
            L10, M10 = (np.sqrt(1+4*a_bar**2 * klm**2)-1)/2, 0
            theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
            phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
            theta1, theta2 = theta1c + deltatheta, theta2c - deltatheta
            phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi
            K0 = klm / np.sqrt(er)

            I10 = spi.quad(partial(R2, l = L10, m = M10, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]

            IdP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[1] * np.sin((theta1+theta1c)/2)**2 * -deltatheta
            IdP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[1] * np.sin((theta2+theta2c)/2)**2 * -deltatheta
            IpP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[0] * deltatheta
            IpP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[0] * deltatheta
            dH2_dr = np.tile(schelkunoff2(l, K0 * b),(m + 1, 1))
            H2 = np.tile(hankel_spher2(l, K0 * b),(m + 1, 1))

            S10 = (((phi2-phi1) * np.sinc(Mm*(phi2-phi1)/(2*np.pi))) ** 2 / (S_lm)) * (np.abs(b*(IdP_1+IdP_2)/dH2_dr)**2 + Mm**2 * np.abs((IpP_1+IpP_2)/(K0*H2))**2)
            S10 = np.dot(delm, S10)
            
            Q10 = (np.pi / 3) * klm * np.sqrt(er) * (b**3 - a**3) * I10 * Dphi / (np.abs(R(np.cos(theta1c),L10,M10,theta1c))**2 * np.sum(S10))
            return tgdel + 1/Qc + 1/Q10
        elif mode == '01':
            L01, M01 = (np.sqrt(1+4*a_bar**2 * klm**2)-1)/2, np.pi/Dphi
            theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
            phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
            theta1, theta2 = theta1c + deltatheta, theta2c - deltatheta
            phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi
            K0 = klm / np.sqrt(er)

            I01 = spi.quad(partial(R2, l = L01, m = M01, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]

            I_th, I_dth = Integrais(theta1, theta2)
            dH2_dr = np.tile(schelkunoff2(l, K0 * b),(m + 1, 1))
            H2 = np.tile(hankel_spher2(l, K0 * b),(m + 1, 1))

            S01 = ((deltaPhi * np.sinc(Mm*deltaPhi/(2*np.pi)) * np.cos(Mm*(phi2-phi1+deltaPhi)/2)) ** 2 / (S_lm)) * (Mm**2 * np.abs(b*I_th/dH2_dr)**2 + np.abs(I_dth/(K0*H2))**2)
            S01 = np.dot(delm, S01)

            Q01 = (np.pi / 24) * klm * np.sqrt(er) * (b**3 - a**3) * I01 * Dphi / (np.abs(R(np.cos((theta1c+theta2c)/2),L01,M01,theta1c))**2 * np.sum(S01))
            return tgdel + 1/Qc + 1/Q01

    def S(Kdes, Dtheta, Dphi):
        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
        theta1, theta2 = theta1c + deltatheta, theta2c - deltatheta
        phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi
        Dphic = phi1 - phi1c
        Kdes = Kdes/np.sqrt(er)

        IdP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[1] * np.sin((theta1+theta1c)/2)**2 * -deltatheta
        IdP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[1] * np.sin((theta2+theta2c)/2)**2 * -deltatheta
        IpP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[0] * deltatheta
        IpP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[0] * deltatheta
        I_th, I_dth = Integrais(theta1, theta2)
        dH2_dr = np.tile(schelkunoff2(l, Kdes * b),(m + 1, 1))
        H2 = np.tile(hankel_spher2(l, Kdes * b),(m + 1, 1))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Sthv = ((1j ** Ml) * (-Kdes * b * sp.lpmn(m, l, 0)[1] * (IdP_1 + IdP_2) / (dH2_dr) \
                + 1j * Mm**2 * sp.lpmn(m, l, 0)[0] * (IpP_1 + IpP_2) / H2 ) * \
                (phi2-phi1) * np.sinc(Mm*(phi2-phi1)/(2*np.pi)) / S_lm)
            Sthv = np.dot(delm, Sthv)
            Sphh = ((1j ** Ml) * (Kdes * Mm**2 * b * I_th * sp.lpmn(m, l, 0)[0] / dH2_dr \
                + 1j * sp.lpmn(m, l, 0)[1] * -I_dth / H2 ) * \
                2 * Dphic * np.sinc(Mm*Dphic/(2 * np.pi)) * np.cos(Mm*(phi2-phi1+Dphic)/2) / S_lm)
            Sphh = np.dot(delm, Sphh)

        Sr = np.sum(Sthv)/np.sum(Sphh)
        return Sr

    def Probe(k01, k10, Kef, Dtheta, Dphi):
        L01, M01 = (np.sqrt(1+4*a_bar**2 * k01**2)-1)/2, np.pi/Dphi
        L10, M10 = (np.sqrt(1+4*a_bar**2 * k10**2)-1)/2, 0
        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c = np.pi/2 - Dphi/2
        theta1, theta2 = theta1c + deltatheta, theta2c - deltatheta

        I01 = spi.quad(partial(R2, l = L01, m = M01, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]
        I10 = spi.quad(partial(R2, l = L10, m = M10, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]

        thetap = np.linspace(theta1, theta2, 101)

        V = (I01 * R(np.cos(theta1c),L10,M10,theta1c))/(2 * I10 * R(np.cos((theta1c+theta2c)/2),L01,M01,theta1c)) * (Kef**2 - k01**2)/(Kef**2 - k10**2) * S(np.real(Kef), Dtheta, Dphi)
        phip = phi1c + np.arccos(R(np.cos(thetap),L10,M10,theta1c) / R(np.cos(thetap),L01,M01,theta1c) * np.abs(V)) / M01
        
        return phip, thetap

    # Início do Projeto
    def imZin(p, Analysis = 0):
        k01 = k10 = 2*np.pi*flm_des*np.sqrt(u0*es)
        Dtheta = Theta_find(k10)[0]
        Dphi = Phi_find(k01, Dtheta)[-1]
        tgef = (tgefTot(k10, '10', Dtheta, Dphi) + tgefTot(k01, '01', Dtheta, Dphi))/2
    
        epsilon = 1
        tol = 1e-4
        steps = 0
        while epsilon > tol:
            steps += 1
            kl = k10 + p*(k01 - k10)
            kll = kl*tgef/2
            phaseK = np.pi/2 - np.angle(S(kl, Dtheta, Dphi))

            k01prev = k01
            k10prev = k10

            # Supondo-se k01 > k10
            k10 = kl - (-1/np.tan(phaseK) + np.sqrt((1/np.tan(phaseK))**2 + 4*(1-p)*p) ) * kll / (2*(1-p))
            k01 = kl + (-1/np.tan(phaseK) + np.sqrt((1/np.tan(phaseK))**2 + 4*(1-p)*p) ) * kll / (2*p)
            
            Dtheta = Theta_find(k10)[0]
            Dphi = Phi_find(k01, Dtheta)[-1]
            tgef = (1-p)*tgefTot(k10, '10', Dtheta, Dphi) + p*tgefTot(k01, '01', Dtheta, Dphi)

            epsilon = np.max([np.abs(k01prev - k01), np.abs(k10prev - k10)])

        Ph, Th = Probe(k01, k10, kl-1j*kll, Dtheta, Dphi)

        Phtheta = interp1d(Th, Ph)

        theta2c = np.pi/2 + Dtheta/2

        def Z50(theta):
            return np.real(Z(flm_des, k01, k10, theta, Phtheta(theta), tgef, Dtheta, Dphi))-50
        root = root_scalar(Z50, bracket=[np.pi/2, theta2c-0.02], method='bisect')

        thetap = np.array(root.root)

        Zin = Z(flm_des, k01, k10, thetap, Phtheta(thetap), tgef, Dtheta, Dphi)

        if  Analysis:
            Dphia = Dphi - 2*deltaPhi
            Dthetaa = Dtheta - 2*deltatheta
            return Dphia, Dthetaa, thetap, Phtheta(thetap), tgefTot(k01, '01', Dtheta, Dphi), tgefTot(k10, '10', Dtheta, Dphi)
        return np.imag(Zin)

    Wrapper = partial(imZin, Analysis = 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        root = root_scalar(Wrapper, bracket=[0.5, 0.7], method='bisect', xtol = 1e-4)
        Dphia, Dthetaa, thetap, phip, tg01, tg10 = imZin(np.array(root.root), Analysis = 1)
    #print('p = ', np.array(root.root))

    parametros = np.array([Dthetaa, Dphia, thetap, phip])
    estimativas = np.array([np.exp(1.5)*df/(2*a_bar*np.sin(thetap)), 0.0, 0.0, tg01, tg10, deltatheta, deltaPhi])
    return parametros, estimativas

#########################################################################################################################################################
# Síntese após atualização dos parâmetros
def synth_ant_pos(flm_des, estim):
    Dphip, Yp, phip_add, tg01, tg10, deltatheta, deltaPhi = estim
    # Constantes gerais
    dtr = np.pi/180         # Graus para radianos (rad/°)
    e0 = 8.854e-12          # (F/m)
    u0 = np.pi*4e-7         # (H/m)
    gamma = 0.577216        # Constante de Euler-Mascheroni
    eps = 1e-5              # Limite para o erro numérico

    # Geometria da cavidade
    a = 100e-3              # Raio da esfera de terra (m)
    h = 1.524e-3            # Espessura do substrato dielétrico (m)
    a_bar = a + h/2         # Raio médio da esfera de terra (m)
    b = a + h               # Raio exterior do dielétrico (m)

    # Coordenadas dos alimentadores coaxiais (rad)
    df = 1.3e-3             # Diâmetro do pino central dos alimentadores coaxiais (m)
    er = 2.55               # Permissividade relativa do substrato dielétrico
    es = e0 * er            # Permissividade do substrato dielétrico (F/m)

    # Funções associadas de Legendre para |z| < 1
    def legP(z, l, u):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = float(u)
            if u.is_integer():
                return np.where(np.abs(z) < 1, (-1)**u * (sp.gamma(l+u+1)/sp.gamma(l-u+1)) * np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1+u,(1-z)/2)  / sp.gamma(u+1), 1)
            else:
                return np.where(np.abs(z) < 1, np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u), 1)

    def legQ(z, l, u):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = float(u)
            if u.is_integer():
                return np.where(np.abs(z) < 1, np.pi * (np.cos((u+l)*np.pi) * np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u) \
            - np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1+z)/2)  / sp.gamma(1-u)) / (2 * np.sin((u+l)*np.pi)), 1)
            else:
                return np.where(np.abs(z) < 1, np.pi * (np.cos(u*np.pi) * np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u) \
            - sp.gamma(l+u+1) * np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1+u,(1-z)/2)  / (sp.gamma(1+u) * sp.gamma(l-u+1))) / (2 * np.sin(u*np.pi)), 1)

    # Derivadas
    def DP(theta, l, u):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = np.cos(theta)
            return l*legP(z, l, u)/np.tan(theta) - (l+u) * legP(z, l-1, u) / np.sin(theta)

    def DQ(theta, l, u):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = np.cos(theta)
            return l*legQ(z, l, u)/np.tan(theta) - (l+u) * legQ(z, l-1, u) / np.sin(theta)

    #########################################################################################################################################################
    # Funções auxiliares de busca de raízes
    def EquationCircTheta(Dtheta, Dphi, m, K):
        theta1, theta2 = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        Lamb = (np.sqrt(1+4*a_bar**2 * K**2)-1)/2
        u = m * np.pi / Dphi
        return np.where(np.abs(np.cos(theta1)) < 1 and np.abs(np.cos(theta2)) < 1, DP(theta1,Lamb,u)*DQ(theta2,Lamb,u)-DQ(theta1,Lamb,u)*DP(theta2,Lamb,u) , 1)

    def EquationCircPhi(Dphi, Dtheta, m, K):
        theta1, theta2 = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        Lamb = (np.sqrt(1+4*a_bar**2 * K**2)-1)/2
        u = m * np.pi / Dphi
        return np.where(np.abs(np.cos(theta1)) < 1 and np.abs(np.cos(theta2)) < 1, DP(theta1,Lamb,u)*DQ(theta2,Lamb,u)-DQ(theta1,Lamb,u)*DP(theta2,Lamb,u) , 1)

    def Theta_find(K, Dphi = 1):
        Thetas = np.linspace(0.1, 359.9, 601) * dtr # Domínio de busca
        roots = []
        Wrapper = partial(EquationCircTheta, Dphi = Dphi,  m = 0, K = K) # Modo TM^r_10 desejado
        k = 0
        for i in range(len(Thetas)-1):
            if Wrapper(Thetas[i]) * Wrapper(Thetas[i+1]) < 0:
                root = root_scalar(Wrapper, bracket=[Thetas[i], Thetas[i+1]], method='bisect')
                roots.append(root.root)
                k += 1

        return np.array(roots)

    def Phi_find(K, Dtheta):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Phis = np.linspace(0.1, 179.9, 351) * dtr # Domínio de busca
            roots = []
            Wrapper = partial(EquationCircPhi, Dtheta = Dtheta,  m = 1, K = K) # Modo TM^r_01 desejado
            k = 0
            for i in range(len(Phis)-1):
                if Wrapper(Phis[i]) * Wrapper(Phis[i+1]) < 0:
                    root = root_scalar(Wrapper, bracket=[Phis[i], Phis[i+1]], method='bisect')
                    roots.append(root.root)
                    k += 1

            return np.array(roots)

    #########################################################################################################################################################
    # Projeto iterativo
    l = m = 35 # m <= l

    Ml, Mm = np.meshgrid(np.arange(0,l+1), np.arange(0,m+1))
    delm = np.ones(m+1)
    delm[0] = 0.5

    def RLC(f, Kex, L, U, thetap, phip, tgef, Dtheta, Dphi):
        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
        Ilm = spi.quad(partial(R2, l = L, m = U, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]

        alpha = h * (R(np.cos(thetap), L, U, theta1c = theta1c) * np.cos(U * (phip-phi1c)) * np.sinc(U * Dphip / (2*np.pi))) * \
            (R(np.cos(thetap), L, U, theta1c = theta1c) * np.cos(U * (phip-phi1c)) * np.sinc(U * Dphip / (2*np.pi))) / ((phi2c - phi1c) * es * a_bar**2 * Ilm)
        if U != 0:
            alpha = 2 * alpha
        
        RLM = alpha/(2 * np.pi * f * tgef)
        CLM = 1/alpha
        LLM = alpha/(2 * np.pi * Kex / (2*np.pi*np.sqrt(u0*es)))**2
        return 1/(1/RLM+1j*(2 * np.pi * f * CLM - 1/(2 * np.pi * f * LLM)))

    def Z(f, k01, k10, thetap, phip, tg01, tg10, Dtheta, Dphi):
        L01, M01 = (np.sqrt(1+4*a_bar**2 * k01**2)-1)/2, np.pi/Dphi
        L10, M10 = (np.sqrt(1+4*a_bar**2 * k10**2)-1)/2, 0
        k = (2 * np.pi * f) * np.sqrt(es * u0) 
        eta = np.sqrt(u0 / es)
        Xp = (eta * k * h / (2 * np.pi)) * (np.log(4 / (k * df)) - gamma)
        return RLC(f, k01, L01, M01, thetap, phip, tg01, Dtheta, Dphi) + RLC(f, k10, L10, M10, thetap, phip, tg10, Dtheta, Dphi) + 1j*Xp*Yp

    def hankel_spher2(n, x, der = 0):
        return sp.riccati_jn(n, x)[der]/x - 1j * sp.riccati_yn(n, x)[der]/x

    def schelkunoff2(n, x, der = 1):
        return sp.riccati_jn(n, x)[der] - 1j * sp.riccati_yn(n, x)[der]

    def Itheta(theta, L, M):
        if M > L:
            return 0
        return sp.lpmn(M, L, np.cos(theta))[0][M, L] * np.sin(theta)

    def IDtheta(theta, L, M):
        if M > L:
            return 0
        return -sp.lpmn(M, L, np.cos(theta))[1][M, L] * np.sin(theta)**2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S_lm = (2 * Ml * (Ml+1) * sp.gamma(1+Ml+Mm)) / ((2*Ml+1) * sp.gamma(1+Ml-Mm))
        S_lm += (1-np.abs(np.sign(S_lm)))*eps

    def Integrais(theta1, theta2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            I_th = np.array([[spi.quad(partial(Itheta, L = i, M = j), theta1 , theta2)[0] for i in range(l+1)] for j in range(m+1)])
            I_dth = np.array([[spi.quad(partial(IDtheta, L = i, M = j), theta1 , theta2)[0] for i in range(l+1)] for j in range(m+1)])
        return I_th, I_dth

    def R(v,L,m, theta1c):
        return (DP(theta1c,L,m)*legQ(v,L,m) - DQ(theta1c,L,m)*legP(v,L,m)) * np.sin(theta1c)

    def R2(v, l, m, theta1c):
        return R(v, l, m, theta1c)**2

    def S(Kdes, Dtheta, Dphi):
        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
        theta1, theta2 = theta1c + deltatheta, theta2c - deltatheta
        phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi
        Dphic = phi1 - phi1c
        Kdes = Kdes/np.sqrt(er)

        IdP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[1] * np.sin((theta1+theta1c)/2)**2 * -deltatheta
        IdP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[1] * np.sin((theta2+theta2c)/2)**2 * -deltatheta
        IpP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[0] * deltatheta
        IpP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[0] * deltatheta
        I_th, I_dth = Integrais(theta1, theta2)
        dH2_dr = np.tile(schelkunoff2(l, Kdes * b),(m + 1, 1))
        H2 = np.tile(hankel_spher2(l, Kdes * b),(m + 1, 1))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Sthv = ((1j ** Ml) * (-Kdes * b * sp.lpmn(m, l, 0)[1] * (IdP_1 + IdP_2) / (dH2_dr) \
                + 1j * Mm**2 * sp.lpmn(m, l, 0)[0] * (IpP_1 + IpP_2) / H2 ) * \
                (phi2-phi1) * np.sinc(Mm*(phi2-phi1)/(2*np.pi)) / S_lm)
            Sthv = np.dot(delm, Sthv)
            Sphh = ((1j ** Ml) * (Kdes * Mm**2 * b * I_th * sp.lpmn(m, l, 0)[0] / dH2_dr \
                + 1j * sp.lpmn(m, l, 0)[1] * -I_dth / H2 ) * \
                2 * Dphic * np.sinc(Mm*Dphic/(2 * np.pi)) * np.cos(Mm*(phi2-phi1+Dphic)/2) / S_lm)
            Sphh = np.dot(delm, Sphh)

        Sr = np.sum(Sthv)/np.sum(Sphh)
        return Sr

    def Probe(k01, k10, Kef, Dtheta, Dphi):
        L01, M01 = (np.sqrt(1+4*a_bar**2 * k01**2)-1)/2, np.pi/Dphi
        L10, M10 = (np.sqrt(1+4*a_bar**2 * k10**2)-1)/2, 0
        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c = np.pi/2 - Dphi/2
        theta1, theta2 = theta1c + deltatheta, theta2c - deltatheta

        I01 = spi.quad(partial(R2, l = L01, m = M01, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]
        I10 = spi.quad(partial(R2, l = L10, m = M10, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]

        thetap = np.linspace(theta1, theta2, 101)

        V = (I01 * R(np.cos(theta1c),L10,M10,theta1c))/(2 * I10 * R(np.cos((theta1c+theta2c)/2),L01,M01,theta1c)) * (Kef**2 - k01**2)/(Kef**2 - k10**2) * S(np.real(Kef), Dtheta, Dphi)
        phip = phi1c + np.arccos(R(np.cos(thetap),L10,M10,theta1c) / R(np.cos(thetap),L01,M01,theta1c) * np.abs(V)) / M01
        
        mask = ~np.isnan(phip)
        phip = phip[mask]
        thetap = thetap[mask]

        return phip, thetap

    # Início do Projeto
    def imZin(p, Analysis = 0):
        k01 = k10 = 2*np.pi*flm_des*np.sqrt(u0*es)
        Dtheta = Theta_find(k10)[0]
        Dphi = Phi_find(k01, Dtheta)[-1]
        tgef = (tg10+tg01)/2
    
        epsilon = 1
        tol = 1e-4
        steps = 0
        while epsilon > tol and steps<20:
            steps += 1
            kl = k10 + p*(k01 - k10)
            kll = kl*tgef/2
            phaseK = np.pi/2 - np.angle(S(kl, Dtheta, Dphi))

            k01prev = k01
            k10prev = k10

            # Supondo-se k01 > k10
            k10 = kl - (-1/np.tan(phaseK) + np.sqrt((1/np.tan(phaseK))**2 + 4*(1-p)*p) ) * kll / (2*(1-p))
            k01 = kl + (-1/np.tan(phaseK) + np.sqrt((1/np.tan(phaseK))**2 + 4*(1-p)*p) ) * kll / (2*p)
            
            Dtheta = Theta_find(k10)[0]
            Dphi = Phi_find(k01, Dtheta)[-1]
            tgef = (1-p)*tg10 + p*tg01

            epsilon = np.max([np.abs(k01prev - k01), np.abs(k10prev - k10)])

        Ph, Th = Probe(k01, k10, kl-1j*kll, Dtheta, Dphi)

        Phtheta = interp1d(Th, Ph)

        def Z50(theta):
            return np.real(Z(flm_des, k01, k10, theta, Phtheta(theta), tg01, tg10, Dtheta, Dphi))-50
        root = root_scalar(Z50, bracket=[np.pi/2, Th[-1]], method='bisect')

        thetap = np.array(root.root)

        Zin = Z(flm_des, k01, k10, thetap, Phtheta(thetap), tg01, tg10, Dtheta, Dphi)

        if  Analysis:
            Dphia = Dphi - 2*deltaPhi
            Dthetaa = Dtheta - 2*deltatheta
            return Dphia, Dthetaa, thetap, Phtheta(thetap)-phip_add
        return np.imag(Zin)

    Wrapper = partial(imZin, Analysis = 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        root = root_scalar(Wrapper, bracket=[0.5, 0.7], method='bisect', xtol = 1e-4)
        Dphia, Dthetaa, thetap, phip = imZin(np.array(root.root), Analysis = 1)
    #print('p = ', np.array(root.root))

    parametros = np.array([Dthetaa, Dphia, thetap, phip])
    return parametros


#print(synth_ant_pos(1575.42e6, [0.02903818, 0.01378051, 0.01301588, 0.01524,    0.01524   ]))
#print(synth_ant_pos(1575.42e6, [0.03300348, 0.01294931, 0.01449906, 0.00850071, 0.00847083]))
#print(synth_ant_pos(1575.42e6, [ 2.89400e-02, -3.10956e+00, -7.00000e-04,  1.98700e-02,  1.65900e-02,  8.43000e-03,  8.33000e-03]))
#print(synth_ant_pos(1575.42e6, [ 2.81861084e-02, -3.03895000e+00, -6.11646776e-04,  1.50885349e-02,  1.80141102e-02,  8.03639301e-03,  7.98419473e-03]))