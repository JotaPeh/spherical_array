import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import os
import time
import warnings
import matplotlib.pyplot as plt

# def calc_3db_angle(gg, angulos):
#     N = len(gg)
#     p = np.argmax(gg)
#     t = gg[p] - 3

#     r = p
#     while gg[(r+1)%N] > t:
#         r = (r+1)%N
#     ar = angulos[(r+1)%N]

#     l = p
#     while gg[(l-1)%N] > t:
#         l = (l-1)%N
#     al = angulos[(l-1)%N]

#     bw = (ar - al) % 360
    
#     return bw
    
def array_synth( amptds, positions, keyDir, isCP = True):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    dtr = np.pi/180         # Degrees to radians (rad/°)

    input_dir = os.path.join("HFSS", keyDir)

    patch_files = {
        "Etheta_re": "rE_Theta_re.csv",
        "Etheta_im": "rE_Theta_im.csv",
        "Ephi_re":   "rE_Phi_re.csv",
        "Ephi_im":   "rE_Phi_im.csv"
    }

    Alphas = positions[0]
    Betas = positions[1]
    Gammas = positions[2]
    
    def load_interpolator(input_dir, filename):
        df = pd.read_csv(os.path.join(input_dir, filename))
        theta_vals = np.sort(df["Theta[deg]"].unique())
        phi_vals = np.sort(df["Phi[deg]"].unique())
        data = np.zeros((len(phi_vals), len(theta_vals)))

        for _, row in df.iterrows():
            phi_idx = np.where(phi_vals == row["Phi[deg]"])[0][0]
            theta_idx = np.where(theta_vals == row["Theta[deg]"])[0][0]
            data[phi_idx, theta_idx] = row.iloc[-1]

        return RegularGridInterpolator((theta_vals, phi_vals), data.T, bounds_error=False, fill_value=None)

    # Load isolated patch pattern
    interp_Etheta_re = load_interpolator(input_dir, patch_files["Etheta_re"])
    interp_Etheta_im = load_interpolator(input_dir, patch_files["Etheta_im"])
    interp_Ephi_re   = load_interpolator(input_dir, patch_files["Ephi_re"])
    interp_Ephi_im   = load_interpolator(input_dir, patch_files["Ephi_im"])

    # Complex field functions
    def E_theta(theta, phi):
        theta, phi = np.broadcast_arrays(theta, phi)
        pts = np.column_stack((theta.ravel(), phi.ravel()))
        result = interp_Etheta_re(pts) + 1j * interp_Etheta_im(pts)
        return result.reshape(theta.shape)

    def E_phi(theta, phi):
        theta, phi = np.broadcast_arrays(theta, phi)
        pts = np.column_stack((theta.ravel(), phi.ravel()))
        result = interp_Ephi_re(pts) + 1j * interp_Ephi_im(pts)
        return result.reshape(theta.shape)

    ############################################################ Rotate Fields

    def rotate(theta, phi, alpha, beta, gamma_y):
        # Rotação em z (alpha), depois em x (beta) depois em y (gamma)
        T = np.array([
            [np.cos(gamma_y), 0, np.sin(gamma_y)],
            [0, 1, 0],
            [-np.sin(gamma_y), 0, np.cos(gamma_y)]
        ]) @ np.array([
            [1, 0, 0],
            [0, np.cos(beta), np.sin(beta)],
            [0, -np.sin(beta), np.cos(beta)]
        ]) @ np.array([
            [np.cos(alpha), np.sin(alpha), 0],
            [-np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]
        ])
        
        A1 = np.array([[np.sin(theta)*np.cos(phi)],[np.sin(theta)*np.sin(phi)],[np.cos(theta)]])
        
        thetan = np.arccos(np.clip(T[2] @ A1, -1.0, 1.0))[0]
        phin = np.arctan2(T[1] @ A1, T[0] @ A1)[0]
        
        A2 = np.array([[np.cos(thetan)*np.cos(phin),np.cos(thetan)*np.sin(phin),-np.sin(thetan)],[-np.sin(phin),np.cos(phin),0]])
        A3 = np.array([[np.cos(theta )*np.cos(phi ),np.cos(theta )*np.sin(phi ),-np.sin(theta )],[-np.sin(phi ),np.cos(phi ),0]]).T
        S = A2 @ T @ A3

        E_original = np.array([E_theta(thetan/dtr, phin/dtr), E_phi(thetan/dtr, phin/dtr)])
        Eg = S.T @ E_original
        return Eg

    phi_vals = np.arange(-180, 181, 1)
    theta_vals = np.arange(0, 181, 1)

    E_g_ph90_pos = []

    for theta in theta_vals:
        for i in range(len(Alphas)):
            alpha = Alphas[i] * dtr
            beta = Betas[i] * dtr
            gamma = Gammas[i] * dtr
            if i == 0:
                Eg_ph90_pos = amptds[i] * rotate(theta * dtr, 90 * dtr, alpha, beta, gamma)
            else:
                Eg_ph90_pos += amptds[i] * rotate(theta * dtr, 90 * dtr, alpha, beta, gamma)
        E_g_ph90_pos.append(Eg_ph90_pos)

    E_g_ph90_pos = np.array(E_g_ph90_pos)

    E_g_ph90_neg = []

    for theta in theta_vals:
        for i in range(len(Alphas)):
            alpha = Alphas[i] * dtr
            beta = Betas[i] * dtr
            gamma = Gammas[i] * dtr
            if i == 0:
                Eg_ph90_neg = amptds[i] * rotate(theta * dtr, -90 * dtr, alpha, beta, gamma)
            else:
                Eg_ph90_neg += amptds[i] * rotate(theta * dtr, -90 * dtr, alpha, beta, gamma)
        E_g_ph90_neg.append(Eg_ph90_neg)

    E_g_ph90_neg = np.array(E_g_ph90_neg)

    E_g_th90 = []

    for phi in phi_vals:
        for i in range(len(Alphas)):
            alpha = Alphas[i] * dtr
            beta = Betas[i] * dtr
            gamma = Gammas[i] * dtr
            if i == 0:
                Eg_th90 = amptds[i] * rotate(90 * dtr, phi * dtr, alpha, beta, gamma)
            else:
                Eg_th90 += amptds[i] * rotate(90 * dtr, phi * dtr, alpha, beta, gamma)
        E_g_th90.append(Eg_th90)

    E_g_th90 = np.array(E_g_th90)

    E_theta_g_th90 = E_g_th90[:, 0]
    E_phi_g_th90   = E_g_th90[:, 1]
    E_theta_g_ph90 = np.concatenate((E_g_ph90_neg[:, 0][::-1], E_g_ph90_pos[:, 0][1:]))
    E_phi_g_ph90   = np.concatenate((E_g_ph90_neg[:, 1][::-1], E_g_ph90_pos[:, 1][1:]))

    angulos = np.arange(-180,181,1)
    # angulos_theta = np.arange(0,181,1)
    # angles = np.arange(0, 360, 30)
    # angles_labels = ['0°', '30°', '60°', '90°', '120°', '150°', '180°', '-150°', '-120°', '-90°', '-60°', '-30°']

    # AR
    th = E_theta_g_th90#[len(E_theta_g_th90) // 4]
    ph = E_phi_g_th90#[len(E_theta_g_th90) // 4]
    kRA = np.abs(th/ph)
    T = np.sqrt(1+kRA**4+2 * kRA**2 * np.cos(2*np.angle(th/ph)))
    RAf = np.sqrt((1+kRA**2+T)/(1+kRA**2-T))
    RAfdB_th90 = np.abs(20*np.log10(np.abs(RAf)))

    th = E_theta_g_ph90#[len(E_theta_g_th90) // 4]
    ph = E_phi_g_ph90#[len(E_theta_g_th90) // 4]
    kRA = np.abs(th/ph)
    T = np.sqrt(1+kRA**4+2 * kRA**2 * np.cos(2*np.angle(th/ph)))
    RAf = np.sqrt((1+kRA**2+T)/(1+kRA**2-T))
    RAfdB_ph90 = np.abs(20*np.log10(np.abs(RAf)))

    if not isCP:
        # 1. E_theta_g (rotacionado) e dados
        E_g_theta = np.abs(E_theta_g_th90)
        E_g_theta_dB = np.clip(20*np.log10(E_g_theta/np.max(E_g_theta)), -30, 0)

        # 2. E_phi_g (rotacionado) e dados
        E_g_phi = np.abs(E_phi_g_th90)
        E_g_phi_dB = np.clip(20*np.log10(E_g_phi/np.max(E_g_phi)), -30, 0)

        # 3. E_theta_g (phi=90, theta varrendo) e dados
        E_g_theta_full = np.abs(E_theta_g_ph90)
        E_g_theta_full_dB = np.clip(20*np.log10(np.abs(E_g_theta_full)/np.max(np.abs(E_g_theta_full))), -30, 0)

        # 4. E_phi_g (phi=90, theta varrendo) e dados
        E_g_phi_full = E_phi_g_ph90
        E_g_phi_full_dB = np.clip(20*np.log10(np.abs(E_g_phi_full)/np.max(np.abs(E_g_phi_full))), -30, 0)

        return angulos, E_g_theta_dB, E_g_phi_dB, E_g_theta_full_dB, E_g_phi_full_dB

    else:
        # RHCP e LHCP para theta = 90°, varrendo phi
        Erhcp = (E_theta_g_th90 + 1j * E_phi_g_th90) / np.sqrt(2)
        Elhcp = (E_theta_g_th90 - 1j * E_phi_g_th90) / np.sqrt(2)

        Erhcp_dB = np.clip(20*np.log10(np.abs(Erhcp)/np.max(np.abs(Erhcp))), -30, 0)
        Elhcp_dB = np.clip(20*np.log10(np.abs(Elhcp)/np.max(np.abs(Elhcp))), -30, 0)

        # Erhcp_dB = 20*np.log10(np.abs(Erhcp))
        # Elhcp_dB = 20*np.log10(np.abs(Elhcp))

        # RHCP e LHCP para phi = 90°, varrendo theta
        Erhcp_full = np.abs((E_theta_g_ph90 + 1j * E_phi_g_ph90) / np.sqrt(2))
        Elhcp_full = np.abs((E_theta_g_ph90 - 1j * E_phi_g_ph90) / np.sqrt(2))

        Erhcp_full_dB = np.clip(20*np.log10(Erhcp_full/np.max(Erhcp_full)), -30, 0)
        Elhcp_full_dB = np.clip(20*np.log10(Elhcp_full/np.max(Elhcp_full)), -30, 0)

        # Erhcp_full_dB = 20*np.log10(Erhcp_full)
        # Elhcp_full_dB = 20*np.log10(Elhcp_full)

        return angulos, Erhcp_dB, Elhcp_dB, Erhcp_full_dB, Elhcp_full_dB, RAfdB_th90, RAfdB_ph90