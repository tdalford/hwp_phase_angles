import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


def make_HWP_transfer_matrix(thicks, thetas, no, ne, wavelength):
    # this code caculats the transfer matrix assumng x-y input and output
    ################
    # thick --- the thickness of each layer, use mm
    # theta --- the angle of the ordinary axis relative to the x axis
    # no    --- the index along the ordinary axis
    # ne    --- the index along the extrodinary axis
    # wavelength  --- the wavelenght, use mm
    ################
    # set up the input layer, assumed to be vacuume
    TF = np.matrix([[1, 0],
                    [0, 1]])
    theta_prev = 0.
    # now iterate through the layers of HWP
    N_layers = np.size(thicks)
    i = 0
    while (i < N_layers):
        d_th = thetas[i] - theta_prev
        Mrot = np.matrix([[np.cos(d_th), np.sin(d_th)],
                          [-np.sin(d_th), np.cos(d_th)]])
        k = 2 * np.pi / wavelength
        phi_o = k * no * thicks[i]
        phi_e = k * ne * thicks[i]
        Mphase = np.matrix([[np.exp(np.complex(0, 1)*phi_o), 0],
                            [0, np.exp(np.complex(0, 1)*phi_e)]])
        TF_layer = np.matmul(Mphase, Mrot)
        TF = np.matmul(TF_layer, TF)
        # now iterate
        theta_prev = thetas[i]
        i += 1
    # now do the last layer
    d_th = 0 - theta_prev
    Mrot = np.matrix([[np.cos(d_th), np.sin(d_th)],
                      [-np.sin(d_th), np.cos(d_th)]])
    TF = np.matmul(Mrot, TF)
    return(TF)


def rotate_HWP(TF, psi):
    Mrot = np.matrix([[np.cos(psi), np.sin(psi)],
                      [-np.sin(psi), np.cos(psi)]])
    out = np.matmul(TF, Mrot)
    out = np.matmul(np.transpose(Mrot), out)
    return(out)


def sim_rot_HWP(TF, nsamps):
    angs = np.arange(nsamps)/nsamps * 2 * np.pi
    Ex_out = np.zeros(nsamps) * np.complex(0, 1)
    Ey_out = np.zeros(nsamps) * np.complex(0, 1)
    i = 0
    while(i < nsamps):
        TF_tmp = rotate_HWP(TF, angs[i])
        Ex_out[i] = TF_tmp[0, 0]
        Ey_out[i] = TF_tmp[1, 0]
        i += 1
    return(angs, Ex_out, Ey_out)


def mod_eff_and_phase(E):
    tmp = np.fft.ifft((np.abs(E))**2)
    mod_eff = 2. * np.abs(tmp[4]) / .5  # / np.abs(tmp[0])
    ##
    tmp = np.arctan2(np.imag(tmp[4]), np.real(tmp[4]))
    phase = 45 + 180/np.pi * tmp/4.
    return(mod_eff, phase)


# compute the fraction of the simualted band that is usable, defined as above a threshold
# def fractional_BW_usable(mod_eff,thresshold):
#    ok = np.where(mod_eff > thresshold )
#    return(np.size(ok)/np.size(mod_eff))

def fractional_BW_usable(mod_eff):
    tmp = np.sum(mod_eff)
    tmp = tmp / np.size(mod_eff)
    return(tmp)


# calcuatle the dispersion in the phase angle
def phase_angle_dispersion(phase_angle, frequency):
    phase_angle = np.unwrap(phase_angle*4.*np.pi/180)/4. * 180/np.pi
    low_band = np.where(frequency < 112.)
    high_band = np.where(frequency > 122.)
    out = np.std(phase_angle[low_band])
    out += np.std(phase_angle[high_band])
    return(out)


def optomization_function(mod_effs, phase_angles, frequency):
    # the bandwidth part, this is multiplitative, with small values being bad
    fBW = fractional_BW_usable(mod_effs)
    alpha = 0.05  # smaller aphpah penalizes more for loss of BW
    # 0.1 means 10% bw loss results in a 70% penalty
    BW_penalty = (((1-fBW)/alpha))
    # the phase constancy part, small is good
    rms_phi = np.sqrt(phase_angle_dispersion(phase_angles, frequency)**2)
    # optomization funciton should get small as things improe
    opt = rms_phi + 2*BW_penalty  # consider addign a thickness penalty
    return(opt)


def optomization_function(mod_effs, phase_angles, frequency):
    # the bandwidth part, this is multiplitative, with small values being bad
    fBW = fractional_BW_usable(mod_effs)
    alpha = 0.05  # smaller aphpah penalizes more for loss of BW
    # 0.1 means 10% bw loss results in a 70% penalty
    BW_penalty = (((1-fBW)/alpha))
    # the phase constancy part, small is good
    rms_phi = np.sqrt(phase_angle_dispersion(phase_angles, frequency)**2)
    # optomization funciton should get small as things improe
    opt = rms_phi + 2*BW_penalty  # consider addign a thickness penalty
    return(opt)


def evaluate_HWP(params):
    # unpack the params
    thicks = np.abs(params[:int(np.size(params)/2)])
    thetas = params[int(np.size(params)/2):]*np.pi/180.
    ne = 3.2 + 0.31/2.
    no = 3.2 - 0.31/2.
    # set up the wavelenght range
    freq_GHZ = np.arange(100) + 70.  # from 70 to 170 GHz
    ok = np.where(np.logical_or(freq_GHZ < 112, freq_GHZ > 122))
    freq_GHZ = freq_GHZ[ok]
    wavelengths = 1./freq_GHZ * 300.  # convert to mm
    # set up the outputs for the calcaulation
    mod_effs = np.zeros(np.shape(wavelengths))
    phases = np.zeros(np.shape(wavelengths))

    # numer of angles to use in the simulations of the HWP
    n_angles = 32

    # make the transfer funciton
    i = 0
    while (i < np.size(wavelengths)):
        TF = make_HWP_transfer_matrix(thicks, thetas, no, ne, wavelengths[i])
        angs, Exo, Eyo = sim_rot_HWP(TF, n_angles)
        mod_eff, phase = mod_eff_and_phase(Exo)
        mod_effs[i] = mod_eff
        phases[i] = phase
        i += 1

    phases = np.unwrap(phases*4.*np.pi/180)/4. * 180/np.pi
    # plots for debugging
    if (np.random.random_sample() < .01):  # very important for speeding it up
        print("thicks: ", thicks)
        print("agnles: ", thetas*180./np.pi)
        plt.plot(freq_GHZ, mod_effs, "b.")
        plt.ylim(0, 1)
        plt.show()
        plt.plot(freq_GHZ, phases-np.mean(phases), "r.")
        plt.show()

    return(optomization_function(mod_effs, phases, freq_GHZ))
