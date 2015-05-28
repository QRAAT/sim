# Test code for position estimation. To run, you'll need the following
# Python packages:
#  utm, numdifftools (available through pip)
#  numpy, scipy, matplotlib 

from qraat.srv import util, signal, position1

import numpy as np
import matplotlib.pyplot as pp

cal_id = 3   # Calibration ID, specifies steering vectors to use. 
dep_id = 105 # Deployment ID, specifies a transmitter. 

# siteID -> UTM position, known positions of sites for source
# localization. The real component is northing, the imaginary
# component is easting. 
sites = {2 : (4261604.51+574239.47j), # site2
         3 : (4261569.32+575013.86j), # site10
         4 : (4260706.17+573882.15j), # site13
         5 : (4260749.75+575321.92j), # site20
         6 : (4260856.82+574794.06j), # site21
         8 : (4261100.56+574000.17j)  # site39
         }
         
# UTM position, initial guess of position.
center = (4260500+574500j) 

zone = (10, 'S') # UTM zone.

# Read steering vectors from file.
db_con = util.get_db('writer')
sv = signal.SteeringVectors(db_con, cal_id)


def real_data():

  # Read signal data, about an hour's worth.
  sv = signal.SteeringVectors.read(cal_id, 'sample/sv')
  sig = signal.Signal.read(sites.keys(), 'sample/sig')
  t_step=120
  t_win=60
  pos = position1.WindowedPositionEstimator(sig, sites, center, sv, 
                             t_step, t_win, method=signal.Signal.Bartlet)
 
  cov = position1.WindowedCovarianceEstimator(pos, sites, max_resamples=100)
  position1.InsertPositionsCovariances(db_con, dep_id, cal_id, zone, pos, cov)

def read_db():
  t_start = 1407452400
  t_end   = 1407455880
  pos = position1.ReadPositions(db_con, dep_id, t_start, t_end)
  conf = position1.ReadConfidenceRegions(db_con, dep_id, t_start, t_end, 0.95)
  bearings = position1.ReadAllBearings(db_con, dep_id, t_start, t_end)


def sim_data():

  # Simpulate signal given known position p.  
  p = center + complex(650,0)
  include = [2,4,6,8]

  sig_n = 0.002 # noise
  rho = signal.scale_tx_coeff(p, 1, sites, include)
  sv_splines = signal.compute_bearing_splines(sv)
  sig = signal.Simulator(p, sites, sv_splines, rho, sig_n, 10, include)
    
  (sig_n, sig_t) = sig.estimate_var()

  pos = position1.PositionEstimator(sig, sites, center, 
                               sv, method=signal.Signal.Bartlet)
  pos.plot('fella.png', 999, sites, center, p)
 
  level=0.95
  E = position1.BootstrapCovariance(pos, sites).conf(level)
  E.display(p)
  #E.plot('conf.png', p)
  
  position1.BootstrapCovariance2(pos, sites).conf(level).display(p)
  #position1.Covariance(pos, sites, p_known=p).conf(level).display(p)


# Testing, testing .... 
#sim_data()
#real_data()
read_db()
