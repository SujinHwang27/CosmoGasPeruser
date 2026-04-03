import numpy as np
from scipy.signal import convolve
from scipy.interpolate import interp1d

# Function to convolve with edge wrapping
def convolve_with_wrap(flux_segment, kernel,nbins):
    extended_flux = np.concatenate((flux_segment[-nbins//2:], flux_segment, flux_segment[:nbins//2]))
    convolved_flux = convolve(extended_flux, kernel, mode='same')
    return convolved_flux[nbins//2:-nbins//2]


def getflux(physics,redshift,NSPEC_FIT=16384,SN=None):
    if physics == 1:
        simname = 'Physics1_nofeedback'
    elif physics == 2:
        simname = 'Physics2_stellarwind'
    elif physics == 3:
        simname = 'Physics3_windAGN'
    elif physics == 4:
        simname = 'Physics4_windstrongAGN'
    else:
        raise ValueError("Physics model must be 1, 2, 3, or 4.")
    
    base = f'./{simname}/'
    filename1 = f'{base}los2048_n{NSPEC_FIT}_z{redshift:.3f}.dat'
    filename2 = f'{base}tauH1_2048_n{NSPEC_FIT}_z{redshift:.3f}.dat'
    
    readdata = open(filename1,"rb")
# Header data
    ztime  = np.fromfile(readdata,dtype=np.double,count=1) # redshift
    omegaM = np.fromfile(readdata,dtype=np.double,count=1) # Omega_m (matter density)
    omegaL = np.fromfile(readdata,dtype=np.double,count=1) # Omega_L (Lambda density)
    omegab = np.fromfile(readdata,dtype=np.double,count=1) # Omega_b (baryon density)
    h100   = np.fromfile(readdata,dtype=np.double,count=1) # Hubble constant, H0 / 100 km/s/Mpc
    box100 = np.fromfile(readdata,dtype=np.double,count=1) # Box size in comoving kpc/h
    Xh     = np.fromfile(readdata,dtype=np.double,count=1) # Hydrogen fraction by mass
    nbins  = np.fromfile(readdata,dtype=np.int32,count=1)  # Number of pixels in each line of sight
    numlos = np.fromfile(readdata,dtype=np.int32,count=1)  # Number of lines of sight

    # Line of sight locations in box (can usually ignore these)
    iaxis  = np.fromfile(readdata,dtype=np.int32,count=numlos[0])  # projection axis, x=1, y=2, z=3
    xaxis  = np.fromfile(readdata,dtype=np.double,count=numlos[0]) # x-coordinate in comoving kpc/h
    yaxis  = np.fromfile(readdata,dtype=np.double,count=numlos[0]) # y-coordinate in comoving kpc/h
    zaxis  = np.fromfile(readdata,dtype=np.double,count=numlos[0]) # z-coordinate in comoving kpc/h

    # Line of sight scale
    posaxis = np.fromfile(readdata,dtype=np.double,count=nbins[0]) # comoving kpc/h
    velaxis = np.fromfile(readdata,dtype=np.double,count=nbins[0]) # km/s

    # Gas density, rho/<rho>
    density = np.fromfile(readdata,dtype=np.double,count=nbins[0]*numlos[0])

    # H1 fraction, fH1 = nH1/nH
    H1frac  = np.fromfile(readdata,dtype=np.double,count=nbins[0]*numlos[0])

    # H1 weighted temperature, K
    temp    = np.fromfile(readdata,dtype=np.double,count=nbins[0]*numlos[0])

    # H1 weighted peculiar velocity, km/s
    vpec    = np.fromfile(readdata,dtype=np.double,count=nbins[0]*numlos[0])

    # Close the binary file
    readdata.close()

    readdata = open(filename2,"rb")
# H1 Ly-alpha optical depth (extracting using LosExtract code)
    tau_H1 = np.fromfile(readdata,dtype=np.double,count=nbins[0]*numlos[0]) # dimensionless
    readdata.close()

    nbins,numlos = nbins[0],numlos[0]
    atime = 1.0 / (1.0 + ztime)
    Hz = 100.0 * h100 * np.sqrt(omegaM / (atime * atime * atime) + omegaL)  # km s^-1 Mpc^-1
    boxkms = Hz * box100 / (1.0e3 * h100 * (1.0 + ztime)) # km/s
    if SN is not None:
        NOISE = 1
        SN_FWHM = SN  # Signal to noise PER RESOLUTION ELEMENT (i.e FWHM) at continuum
    else:
        NOISE = 0
    FWHM = 7.0 #19.0  # Approximate FWHM.

    lambda_lya_rest = 1215.6701  # Angstroms
    ckms = 2.99792458e5 # Speed of light in km/s


    if ztime < 0.5:
        COS_LSF = 1
        COS_PIXEL = 0.00997  # Pixel size in Angstrom for G130M
    else:
        COS_LSF = 0

    if ztime < 1.7:
        RESCALE = 0
    else:
        RESCALE = 1  # rescale to match observed mean flux

    if RESCALE == 1:
        # Kim et al. 2007
        taueff = 0.0023 * (1.0 + ztime)**3.65
        
        flux_obs = np.exp(-1.0 * taueff)

        scale = 0.01
        for iscale in range(1000):
            scale_old = scale
            scale = scale +  ( np.mean(np.exp(-scale * tau_H1)) - flux_obs ) / np.mean(tau_H1 * np.exp(-scale * tau_H1))
            dscale = abs(scale - scale_old)
            if dscale < 1.0e-4:
                break
        
        flux = np.exp(-1.0 * scale * tau_H1)
        
    else:
        flux = np.exp(-1.0 * tau_H1)

    # STEP 2: Convolve with instrument profile

    if COS_LSF == 1:
        # --- Load LSF data ---
        wavindex = 31  # wavelength index
        fileLSF = './COS_LSF1_cw1327_75_z0.100.dat'
        data = np.loadtxt(fileLSF)
        
        # --- Calculate redshift of LSF wavelength ---
        zLya_LSF = data[wavindex, 0] / lambda_lya_rest - 1.0
        

        print(fileLSF)
        print(f'Using COS LSF at wavelength [A], zLya_LSF: {data[wavindex, 0]}, {zLya_LSF}')
        
        # --- Extract LSF profile ---
        LSF = data[wavindex, 1:]
        n_lsf = len(LSF)
        LSFpix = np.arange(n_lsf) - (n_lsf - 1)/2  # Centered pixel indices
        
        # --- Calculate velocity scale for LSF ---
        wvel = velaxis[:nbins] / ckms
        lambda_sim = lambda_lya_rest * (1.0 + ztime) * np.sqrt((1.0 + wvel)/(1.0 - wvel))
        
        nbins_LSF = int((lambda_sim.max() - lambda_sim.min()) / COS_PIXEL)
        dlogl_LSF = (np.log10(lambda_sim.max()) - np.log10(lambda_sim.min())) / (nbins_LSF - 1)
        
        loglambda_LSF = np.log10(lambda_sim.min()) + dlogl_LSF * np.arange(nbins_LSF)
        lr = 10**loglambda_LSF / (lambda_lya_rest * (1 + ztime))
        velaxis_LSF = ckms * (lr**2 - 1.0) / (lr**2 + 1.0)
        dv_LSF = velaxis_LSF[nbins_LSF//2] - velaxis_LSF[nbins_LSF//2 - 1]
        
        # --- Create LSF kernel ---
        LSFkms = LSFpix * dv_LSF
        xx_kms = boxkms * (np.arange(nbins) - nbins/2.0) / nbins
        
        # Interpolate in log space
        interp_func = interp1d(LSFkms, np.log10(LSF), 
                            bounds_error=False, 
                            fill_value=-np.inf)  # Handle out-of-bounds
        log_kernel = interp_func(xx_kms)
        kernel = 10**log_kernel
        
        # Set values outside LSF range to zero
        kernel[np.abs(xx_kms) > np.max(LSFkms)] = 0.0
        
        # Normalize kernel
        kernel /= np.sum(kernel)
        
        for i in range(numlos):
            start = i * nbins
            end = (i + 1) * nbins
            flux_segment = flux[start:end]
            
            # Convolve with wrapping edges
            flux[start:end] = convolve_with_wrap(flux_segment, kernel,nbins)

    else:        
        FWHMrel = FWHM * nbins / boxkms
        sigma = FWHMrel / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        # Create the Gaussian kernel
        xx = np.arange(nbins) - nbins / 2.0
        kernel = 1.0 / (sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (xx / sigma) ** 2)

        # Normalize the kernel
        kernel /= np.sum(kernel)

        # Assuming flux is your input array
        for i in range(numlos):
            start = i * nbins
            end = (i + 1) * nbins
            flux_segment = flux[start:end]
            
            # Convolve with wrapping edges
            flux[start:end] = convolve_with_wrap(flux_segment, kernel,nbins)


    wvel = velaxis[0:nbins] / ckms
    lambda_lya = lambda_lya_rest * (1.0 + ztime) * np.sqrt((1.0 + wvel) / (1.0 - wvel))
    loglambda = np.log10(lambda_lya)


    lr = (10.0)**loglambda/(lambda_lya_rest*(1.0+ztime))
    velax = ckms*(lr*lr - 1.0)/(lr*lr + 1.0)

    shift_index = np.zeros(numlos, dtype=int)
    for i in range(numlos):
        ind = np.where(flux[i*nbins : (i+1)*nbins] == np.max(flux[i*nbins : (i+1)*nbins]))
        # shift_index[i] = ind[0][0]
        shift_index[i] = 0

        flux[i*nbins : (i+1)*nbins] = np.roll(flux[i*nbins : (i+1)*nbins], -shift_index[i])

    dv = velax[nbins // 2] - velax[nbins // 2 - 1]
    


    if NOISE == 1:
        for i in range(numlos):
            rand1 = np.random.uniform(size=nbins)
            rand2 = np.random.uniform(size=nbins)
            SNpix = SN_FWHM / np.sqrt(FWHM / dv)
            sigpix = 1.0 / SNpix
            rvg = np.sqrt(-2.0 * np.log(1.0 - rand1)) * np.sin(2.0 * np.pi * rand2) * sigpix

            flux[i * nbins:(i + 1) * nbins] += rvg

    arr2d_flux = np.zeros((NSPEC_FIT, nbins))
    axis = np.zeros((NSPEC_FIT, 4))

    for i in range(NSPEC_FIT):
        arr2d_flux[i, :] = flux[i*nbins:(i+1)*nbins]
        axis[i, :] = [iaxis[i], xaxis[i], yaxis[i], zaxis[i]]
    return arr2d_flux, axis, velax, nbins, numlos

# _, axis, _, nbins, numlos = getflux(1, 0.3)
# print(f"axis:{axis.shape}, \n {nbins}, {numlos}")