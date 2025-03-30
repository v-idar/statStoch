import numpy as np
from matplotlib import pyplot as plt

import scipy.signal
import librosa as libr

def signalsim():
    """
    signalsim() simulates a sum of harmonic stochastic process
    
    :return: x,t where x is the data and t is the time vector
    """ 
    # Parameter values
    N = 500  # Number of samples in the realization
    f1, f2 = 10, 20  # Frequencies of the harmonic process
    sigma1, sigma2 = 2, 2  # Parameters for Rayleigh distributed amplitudes
    fs = 256  # Sample frequency

    # Create data
    n = np.arange(N)  # Time sample vector
    A1, A2 = np.random.rayleigh(scale=sigma1), np.random.rayleigh(scale=sigma2)  # Rayleigh distributed amplitudes
    phi1, phi2 = 2 * np.pi * np.random.rand(), 2 * np.pi * np.random.rand()  # Uniformly distributed phases 0 to 2pi

    # Simulated data sequence
    x = A1 * np.cos(2 * np.pi * f1 * n / fs + phi1) + A2 * np.cos(2 * np.pi * f2 * n / fs + phi2)

    # Time vector relating the sample vector to a time scale corresponding to the sampling frequency
    t = n / fs
    return x,t

def getCov(x, max_lag, r_or_rho):
    """
    getCov Calculates the covariance function or correlation function of data x.

    :param x: data vector
    :param max_lag: Specfify the max lag to calculate the covariance for
    :param r_or_rho: identify if covariance ("r") or correlation ("rho") is sought
    :return: r, lags where r is the covariance and lags are the lags
    """ 
    r = scipy.signal.correlate(x, x) / len(x)
    lags = scipy.signal.correlation_lags(len(x), len(x))
    r = r[np.abs(lags)<=max_lag]
    lags = lags[np.abs(lags)<=max_lag]
    if r_or_rho == "r":
        return r, lags
    return r/np.max(r), lags

def melFil(fs,nfft):
    """
    melFil plots and gives mel filters

    :param fs: sample frequency
    :param nfft: number of fft points
    :return: h where h are the mel filters 
    """ 
    h = libr.filters.mel(sr=fs, n_fft=nfft,n_mels=32)
    plt.figure()
    plt.plot(np.arange(len(h.T))/len(h.T)*fs/2,h.T)
    return h
    
def melSpectrogram(S,fs,hop_length,nfft):
    """
    melSpectrogram plots and gives mel filters

    :param S: the spectrogram calculated of data
    :param fs: sample frequency
    :param hop_length: set as noverlap
    :param nfft: number of fft points
    :return: _
    """   
    mel_signal = libr.feature.melspectrogram(S=S, sr=fs,n_mels=32)
    spectrogram = np.abs(mel_signal)
    power_to_db = libr.power_to_db(spectrogram, ref=np.max)
    plt.figure(figsize=(8, 7))
    libr.display.specshow(power_to_db, sr=fs, x_axis="time", y_axis="mel", cmap="magma", hop_length=hop_length)
    plt.colorbar(label="dB")
    plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
    plt.xlabel('Time', fontdict=dict(size=15))
    plt.ylabel('Frequency', fontdict=dict(size=15))
    plt.show()

def poles_and_zeros(C,A):
    """
    poles_and_zeros for polynomials, give the roots 
    
    :param C: Polynomial of MA part
    :param A: Polynomial of AR part
    :return: n,p the zeros and poles respectively.
    """  
    if len(A):
        p = np.roots(A)
    if len(C):
        n = np.roots(C)
    return n,p

def plot_poles_and_zeros(n,p,ax):
    """
    plot_poles_and_zeros plot the zeros and roots 
    
    :param C: zeros
    :param A: roots
    :return: _
    """
    ax.scatter(np.real(p),np.imag(p),c='r', marker='x')
    ax.scatter(np.real(n),np.imag(n), c='b', marker='o')
    theta = np.linspace( 0 , 2 * np.pi , 200)
    radius = 1
    a = radius * np.cos( theta )    
    b = radius * np.sin( theta )
    ax.plot( a, b,c='k')
    ax.set_aspect('equal')
    ax.set_ylabel("Imaginary Part")
    ax.set_xlabel("Real Part")
    
    
def calculate_pmtm(x, nfft=1024, NW=1):
    """
    Calculates the Pseudo-Multitaper Method (PMTM) spectrum.

    Args:
        x (numpy.ndarray): data.
        nfft (int, optional): Number of points for the FFT. Defaults to 1024.
        K (int, optional): Number of tapers. Defaults to 1.
        NW (float, optional): Time-half bandwidth product. Defaults to 1.

    Returns:
        numpy.ndarray: PMTM spectrum.
    """
    K = 2*NW-1
    v = scipy.signal.windows.dpss(len(x), NW, Kmax=K)
    pmtm = np.zeros([nfft, K])
    for i in range(K):
        X = np.fft.fft(v[i, :].T * x, nfft)
        pmtm[:, i] = X * np.conj(X)
    
    return np.mean(pmtm[:int(nfft/2+1),:], axis=1)


def arcov(x, p):
    """
    Estimate AR parameters using the covariance method.

    Args:
        x (numpy.ndarray): Input array (vector or matrix).
        p (int): Model order (positive integer).

    Returns:
        tuple: AR coefficients (a) and white noise variance estimate (e).
    """
    
    # Initialize output arrays
    a = np.zeros([p,1])

    # Generate the appropriate data matrix
    U = x[:-p,np.newaxis]
    for i in range(1,p):
        U = np.hstack([x[i:-(p-i),np.newaxis],U])
    U = -U
    x = x[p:,np.newaxis]
    # Estimate the parameters of the AR
    theta = np.linalg.pinv(U.T @ U) @ U.T @ x

    # Estimate the input white noise variance
    e = (x-U @ theta).T @ (x-U @ theta) / (len(x)-p)
    
    return np.vstack((1,theta)), e[0][0]

def completeAR(s,fs,p):
    """
    completeAR Given the sound vector s calculate a AR(p) approximation scheme for reconstructing the sound.   
    
    :param s: sound data
    :param fs: sample frequency
    :param p: Order of AR model
    :return: datarec, fsx where datarec is the reconstructed sound and fsx is the output sample frequency
    """
    xtotal = scipy.signal.decimate(s,6)  # Decimate the data
    fsx = fs / 6 # Reduced sampling frequency

    ndata = len(xtotal)
    t = np.arange(ndata) / fsx

    tsec = 20 * 10**(-3)  # Set length of time-windows (20 ms)
    n = int(round(fsx * tsec))  # Sample length of each time-window (160 samples)
    nosec = ndata // n  # Number of sections with length 20ms

    arp_mat = np.zeros((nosec, p+1))
    sigma2_v = np.zeros(nosec)
    for i in range(nosec):
        x = xtotal[i*n:(i+1)*n]  # Pick out the i:th 20 ms section
        arp, sigma2 = arcov(x, p)  # Estimate the AR-model
        arp_mat[i, :] = arp[:,0]  # Store the AR-parameters as row vectors
        sigma2_v[i] = sigma2

    # Reconstruct the whole sequence in each time frame using the AR parameters
    datarec = np.zeros(ndata)
    for i in range(nosec):
        datarec[i*n:(i+1)*n] = scipy.signal.lfilter([np.sqrt(sigma2_v[i])], arp_mat[i, :].tolist(),  np.random.randn(n,1),axis=0)[:,0]

    # Plot original and reconstructed sound
    plt.figure()
    plt.subplot(211)
    plt.plot(t, xtotal,linewidth=0.1)
    plt.title('Original sound')
    plt.xlabel('Time (s)')
    plt.subplot(212)
    plt.plot(t, datarec,linewidth=0.1)
    plt.title('Reconstructed sound')
    plt.xlabel('Time (s)')
    plt.tight_layout() # Adjusts grid to make plots always fit.
    plt.show()
    
    return datarec,fsx
