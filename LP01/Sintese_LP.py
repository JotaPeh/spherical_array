from scipy import special as sp
from scipy.optimize import root_scalar
from scipy import integrate as spi
import numpy as np
from functools import partial
import warnings

def synth_ant(flm_des, modo):
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

    delt = h/a       # Largura angular do campo de franjas

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

    def EquationCircPhi(Dphi, m, K):
        Dtheta = 1.3*Dphi
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

    def Phi_find(K):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Phis = np.linspace(0.1, 179.9, 351) * dtr # Domínio de busca
            roots = []
            Wrapper = partial(EquationCircPhi,  m = 1, K = K) # Modo TM^r_01 desejado
            k = 0
            for i in range(len(Phis)-1):
                if Wrapper(Phis[i]) * Wrapper(Phis[i+1]) < 0:
                    root = root_scalar(Wrapper, bracket=[Phis[i], Phis[i+1]], method='bisect')
                    roots.append(root.root)
                    k += 1

            return np.array(roots)


    def Equation(l, theta1f, theta2f, m, phi1f, phi2f):
        u = m * np.pi / (phi2f - phi1f)
        return np.where(np.abs(np.cos(theta1f)) < 1 and np.abs(np.cos(theta2f)) < 1, DP(theta1f,l,u)*DQ(theta2f,l,u)-DQ(theta1f,l,u)*DP(theta2f,l,u) , 1)

    def root_find(m, theta1c, theta2c, phi1c, phi2c):
        Lambda = np.linspace(-0.1, 32, 64)
        rootsLambda = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            roots = []
            Wrapper = partial(Equation, theta1f = theta1c, theta2f = theta2c, m = m, phi1f = phi1c, phi2f = phi2c)
            k = 0
            for i in range(len(Lambda)-1):
                if Wrapper(Lambda[i]) * Wrapper(Lambda[i+1]) < 0:
                    root = root_scalar(Wrapper, bracket=[Lambda[i], Lambda[i+1]], method='bisect')
                    roots.append(root.root)
                    k += 1

            # Remoção de raízes inválidas
            k = 0
            for r in roots:
                if r < m * np.pi / (phi2c - phi1c) - 1 + eps and round(r, 5) != 0:
                    k += 1
            rootsLambda.append(roots[k:k+5])
        return rootsLambda

    #########################################################################################################################################################
    # Projeto iterativo
    def RLC(f, Kex, L, U, thetap, phip, Dphip, tgef, Dtheta, Dphi):
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

    def Z(f, klm, thetap, phip, Dtheta, Dphi, m_mode):
        tgef = tgefTot(klm, Dtheta, Dphi, m_mode)
        Dphip = np.exp(1.5)*df/(2*a_bar*np.sin(thetap))
        L, M = (np.sqrt(1+4*a_bar**2 * klm**2)-1)/2, m_mode*np.pi/Dphi
        k = (2 * np.pi * f) * np.sqrt(es * u0) 
        eta = np.sqrt(u0 / es)
        Xp = (eta * k * h / (2 * np.pi)) * (np.log(4 / (k * df)) - gamma)
        return RLC(f, klm, L, M, thetap, phip, Dphip, tgef, Dtheta, Dphi) + 1j*Xp

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

    l = m = 15
    Ml, Mm = np.meshgrid(np.arange(0,l+1), np.arange(0,m+1))
    delm = np.ones(m+1)
    delm[0] = 0.5

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

    def tgefTot(klm, Dtheta, Dphi, m_mode):
        Rs = np.sqrt(klm * np.sqrt(u0/es) / (2 * sigma))
        Qc = klm * np.sqrt(u0/es) * h / (2 * Rs)
        Qc = Qc * (3*a**2 + 3*a*h + h**2) / (3*a**2 + 3*a*h + h**2 * 3/2)

        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
        theta1, theta2 = theta1c + delt, theta2c - delt
        phi1, phi2 = phi1c + delt, phi2c - delt
        K0 = klm / np.sqrt(er)

        L, M = (np.sqrt(1+4*a_bar**2 * klm**2)-1)/2, m_mode*np.pi/Dphi
        I = spi.quad(partial(R2, l = L, m = M, theta1c = theta1c), np.cos(theta2c), np.cos(theta1c))[0]
        H2 = np.tile(hankel_spher2(l, K0 * b),(m + 1, 1))

        if m_mode == 0:
            IdP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[1] * np.sin((theta1+theta1c)/2)**2 * -delt
            IdP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[1] * np.sin((theta2+theta2c)/2)**2 * -delt
            IpP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[0] * delt
            IpP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[0] * delt
            dH2_dr = np.tile(schelkunoff2(l, K0 * b),(m + 1, 1))

            S10 = (((phi2-phi1) * np.sinc(Mm*(phi2-phi1)/(2*np.pi))) ** 2 / (4*S_lm)) * (np.abs(b*(IdP_1+IdP_2)/dH2_dr)**2 + Mm**2 * np.abs((IpP_1+IpP_2)/(K0*H2))**2)
            S10 = np.sum(np.dot(delm, S10))

            Q10 = (np.pi / 12) * klm * np.sqrt(er) * (b**3 - a**3) * I * Dphi / (np.abs(R(np.cos(theta1c),L,M,theta1c))**2 * S10)
            return tgdel + 1/Qc + 1/Q10

        elif m_mode == 1:
            I_th, I_dth = Integrais(theta1, theta2)
            dH2_dr = np.tile(schelkunoff2(l, K0 * b),(m + 1, 1))

            S01 = ((delt * np.sinc(Mm*delt/(2*np.pi)) * np.cos(Mm*(phi2-phi1+delt)/2)) ** 2 / (4*S_lm)) * (Mm**2 * np.abs(b*I_th/dH2_dr)**2 + np.abs(I_dth/(K0*H2))**2)
            S01 = np.sum(np.dot(delm, S01))

            Q01 = (np.pi / 96) * klm * np.sqrt(er) * (b**3 - a**3) * I * Dphi / (np.abs(R(np.cos((theta1c+theta2c)/2),L,M,theta1c))**2 * S01)
            return tgdel + 1/Qc + 1/Q01

    # Início do Projeto
    def LinProj(mode = '10'):
        if mode != 0 and mode != 1 and mode != 10 and mode != '10' and mode != '01'  :
            print('Modo Inválido!')
            exit()
        flm_new = flm_des
        m_mode = int(mode) % 10
        klm = 2*np.pi*flm_des*np.sqrt(u0*es)
        phip = thetap = np.pi/2

        if m_mode == 0:
            Dtheta = Theta_find(klm)[0]
            Dphi = 1.3*Dtheta
        elif m_mode == 1:
            Dphi = Phi_find(klm)[-4] # Cuidado! O -4 não é validado em geral, carece de mais testes e seleção de raízes corretas
            Dtheta = 1.3*Dphi

        epsilon = 1
        tol = 1e-4
        steps = 0
        while epsilon > tol:
            steps += 1
            if m_mode == 0:
                def Z50(theta):
                    return np.real(Z(flm_des, klm, theta, phip, Dtheta, Dphi, m_mode))-50
                root = root_scalar(Z50, bracket=[np.pi/2, np.pi/2+Dtheta/2-0.02-h/a], method='bisect')
                thetap = np.array(root.root)
            elif m_mode == 1:
                def Z50(phi):
                    reZ = np.real(Z(flm_des, klm, thetap, phi, Dtheta, Dphi, m_mode))-50
                    return reZ
                root = root_scalar(Z50, bracket=[np.pi/2, np.pi/2+Dphi/2-0.02-h/a], method='bisect')
                phip = np.array(root.root)

            def Z0(f):
                return np.imag(Z(f, klm, thetap, phip, Dtheta, Dphi, m_mode))
            root = root_scalar(Z0, bracket=[0.9*flm_des, 1.02*flm_des], method='bisect')
            flm_new = np.array(root.root)

            if m_mode == 0:
                Dtheta *= flm_new/flm_des
                Dphi = 1.3 * Dtheta
            elif m_mode == 1:
                Dphi *= flm_new/flm_des
                Dtheta = 1.3 * Dphi
            
            theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
            phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2

            lbd = root_find(m_mode, theta1c, theta2c, phi1c, phi2c)[0][1-m_mode]
            klm = np.sqrt(lbd*(lbd+1))/a_bar

            Zin = Z(flm_des, klm, thetap, phip, Dtheta, Dphi, m_mode)
            epsilon = np.abs(np.max([np.imag(Zin),np.real(Zin)-50.0]))
            
            Dphia = Dphi - 2*delt
            Dthetaa = Dtheta - 2*delt
        return Dphia, Dthetaa, thetap, phip, tgefTot(klm, Dtheta, Dphi, m_mode)

    Dphia, Dthetaa, thetap, phip, tgef = LinProj(modo)

    parametros = np.array([Dthetaa, Dphia, thetap, phip])
    estimativas = np.array([np.exp(1.5)*df/(2*a_bar*np.sin(thetap)), tgef, h/a])
    return parametros, estimativas

#########################################################################################################################################################
# Síntese após atualização dos parâmetros
def synth_ant_pos(flm_des, modo, estim):
    Dphip, tgef, delt = estim
    
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

    def EquationCircPhi(Dphi, m, K):
        Dtheta = 1.3*Dphi
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

    def Phi_find(K):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Phis = np.linspace(0.1, 179.9, 351) * dtr # Domínio de busca
            roots = []
            Wrapper = partial(EquationCircPhi,  m = 1, K = K) # Modo TM^r_01 desejado
            k = 0
            for i in range(len(Phis)-1):
                if Wrapper(Phis[i]) * Wrapper(Phis[i+1]) < 0:
                    root = root_scalar(Wrapper, bracket=[Phis[i], Phis[i+1]], method='bisect')
                    roots.append(root.root)
                    k += 1

            return np.array(roots)

    def Equation(l, theta1f, theta2f, m, phi1f, phi2f):
        u = m * np.pi / (phi2f - phi1f)
        return np.where(np.abs(np.cos(theta1f)) < 1 and np.abs(np.cos(theta2f)) < 1, DP(theta1f,l,u)*DQ(theta2f,l,u)-DQ(theta1f,l,u)*DP(theta2f,l,u) , 1)

    def root_find(m, theta1c, theta2c, phi1c, phi2c):
        Lambda = np.linspace(-0.1, 32, 64)
        rootsLambda = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            roots = []
            Wrapper = partial(Equation, theta1f = theta1c, theta2f = theta2c, m = m, phi1f = phi1c, phi2f = phi2c)
            k = 0
            for i in range(len(Lambda)-1):
                if Wrapper(Lambda[i]) * Wrapper(Lambda[i+1]) < 0:
                    root = root_scalar(Wrapper, bracket=[Lambda[i], Lambda[i+1]], method='bisect')
                    roots.append(root.root)
                    k += 1

            # Remoção de raízes inválidas
            k = 0
            for r in roots:
                if r < m * np.pi / (phi2c - phi1c) - 1 + eps and round(r, 5) != 0:
                    k += 1
            rootsLambda.append(roots[k:k+5])
        return rootsLambda

    #########################################################################################################################################################
    # Projeto iterativo
    def RLC(f, Kex, L, U, thetap, phip, Dphip, tgef, Dtheta, Dphi):
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

    def Z(f, klm, thetap, phip, Dtheta, Dphi, m_mode):
        L, M = (np.sqrt(1+4*a_bar**2 * klm**2)-1)/2, m_mode*np.pi/Dphi
        k = (2 * np.pi * f) * np.sqrt(es * u0) 
        eta = np.sqrt(u0 / es)
        Xp = (eta * k * h / (2 * np.pi)) * (np.log(4 / (k * df)) - gamma)
        return RLC(f, klm, L, M, thetap, phip, Dphip, tgef, Dtheta, Dphi) + 1j*Xp

    l = m = 15
    Ml, Mm = np.meshgrid(np.arange(0,l+1), np.arange(0,m+1))
    delm = np.ones(m+1)
    delm[0] = 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S_lm = (2 * Ml * (Ml+1) * sp.gamma(1+Ml+Mm)) / ((2*Ml+1) * sp.gamma(1+Ml-Mm))
        S_lm += (1-np.abs(np.sign(S_lm)))*eps

    def R(v,L,m, theta1c):
        return (DP(theta1c,L,m)*legQ(v,L,m) - DQ(theta1c,L,m)*legP(v,L,m)) * np.sin(theta1c)

    def R2(v, l, m, theta1c):
        return R(v, l, m, theta1c)**2

    # Início do Projeto
    def LinProj(mode = '10'):
        if mode != 0 and mode != 1 and mode != 10 and mode != '10' and mode != '01'  :
            print('Modo Inválido!')
            exit()
        flm_new = flm_des
        m_mode = int(mode) % 10
        klm = 2*np.pi*flm_des*np.sqrt(u0*es)
        phip = thetap = np.pi/2

        if m_mode == 0:
            Dtheta = Theta_find(klm)[0]
            Dphi = 1.3*Dtheta
        elif m_mode == 1:
            Dphi = Phi_find(klm)[-4] # Cuidado! O -4 não é validado em geral, carece de mais testes e seleção de raízes corretas
            Dtheta = 1.3*Dphi

        epsilon = 1
        tol = 1e-4
        steps = 0
        while epsilon > tol:
            steps += 1
            if m_mode == 0:
                def Z50(theta):
                    return np.real(Z(flm_des, klm, theta, phip, Dtheta, Dphi, m_mode))-50
                root = root_scalar(Z50, bracket=[np.pi/2, np.pi/2+Dtheta/2-0.02-h/a], method='bisect')
                thetap = np.array(root.root)
            elif m_mode == 1:
                def Z50(phi):
                    reZ = np.real(Z(flm_des, klm, thetap, phi, Dtheta, Dphi, m_mode))-50
                    return reZ
                root = root_scalar(Z50, bracket=[np.pi/2, np.pi/2+Dphi/2-0.02-h/a], method='bisect')
                phip = np.array(root.root)

            def Z0(f):
                return np.imag(Z(f, klm, thetap, phip, Dtheta, Dphi, m_mode))
            root = root_scalar(Z0, bracket=[0.9*flm_des, 1.02*flm_des], method='bisect')
            flm_new = np.array(root.root)

            if m_mode == 0:
                Dtheta *= flm_new/flm_des
                Dphi = 1.3 * Dtheta
            elif m_mode == 1:
                Dphi *= flm_new/flm_des
                Dtheta = 1.3 * Dphi
            
            theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
            phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2

            lbd = root_find(m_mode, theta1c, theta2c, phi1c, phi2c)[0][1-m_mode]
            klm = np.sqrt(lbd*(lbd+1))/a_bar

            Zin = Z(flm_des, klm, thetap, phip, Dtheta, Dphi, m_mode)
            epsilon = np.abs(np.max([np.imag(Zin),np.real(Zin)-50.0]))
            
            Dphia = Dphi - 2*delt
            Dthetaa = Dtheta - 2*delt
        return Dphia, Dthetaa, thetap, phip

    Dphia, Dthetaa, thetap, phip = LinProj(modo)

    parametros = np.array([Dthetaa, Dphia, thetap, phip])
    return parametros