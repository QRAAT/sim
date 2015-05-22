# beacon.dat -- Position and covariance estimates of all available beacon data
# beacon-58.dat -- Exclude sites 5 and 8.
# beacon-2458.dat -- Exclude sites 2, 4, 5, and 8.  
# beacon-2458n.dat -- Exclude sites 2, 4, 5, and 8, normalized 

from qraat.srv import util, signal, position1

import scipy.stats
import numpy as np
import matplotlib.pyplot as pp
import pickle

suffix = 'dat'

position1.NORMALIZE_SPECTRUM=False

cal_id = 3
t_step = 15
t_win = 15
t_chunk = 3600 / 4 
conf_level=0.95

dep_id = 60
t_start = 1383098400.514320
t_end = 1383443999.351099
#fn = 'beacon'
#fn = 'beacon-58'
fn = 'beacon-2458'
#fn = 'beacon-2458n'


#dep_id = 61
#t_start = 1396725598.548015
#t_end = 1396732325.777558
#fn = 'walking'

# Location of site34. 
site34 = np.complex(4260910.87, 574296.45)


def process(sv):  

  ct = 1; good = 0; total = 0
  P = {} # site_id's --> position estimate
  C = {} # site_id's --> confidence region estimate
  for t in np.arange(t_start, t_end+t_chunk, t_chunk):
      
    print "chunk", ct; ct+=1

    # Signal data
    sig = signal.Signal(db_con, dep_id, t, t+t_chunk, exclude=[2,4,5,8])
    if sig.t_start == float("+inf"):
      continue

    # Compute positions
    positions = position1.WindowedPositionEstimator(dep_id, sites, site34, sig, sv, 
                             t_step, t_win, method=signal.Signal.Bartlet)
   
    for pos in positions:

      print pos.splines.keys(), 
      site_ids = tuple(set(pos.splines.keys()))
      if P.get(site_ids) is None:
        P[site_ids] = []
        C[site_ids] = []
      
      #pos.plot('fella', sites, center, p_known=site34)
      #assert False

      P[site_ids].append(pos.p)
      
      if pos.p is not None:
        try: 
          cov = position1.BootstrapCovariance(pos, sites, max_resamples=500)
          E = cov.conf(conf_level)
          C[site_ids].append((E.angle, E.axes[0], E.axes[1]))
          print "Ok"
          good += 1
        except np.linalg.linalg.LinAlgError: 
          C[site_ids].append(None)
          print "non positive indefinite"
        except position1.PosDefError: 
          C[site_ids].append(None)
          print "non positive indefinite (PosDefError)"
        except position1.BootstrapError:
          C[site_ids].append(None)
          print "samples ..." 
      else: C[site_ids].append(None)
      total += 1

  print 'good', good, "out of", total, "(t_win=%d, norm=%s)" % (
                              t_win, position1.NORMALIZE_SPECTRUM)
  return (P, C)



def plot_map(p_known, sites, fn): 
  fig = pp.gcf()
  ax = fig.add_subplot(111)
  ax.axis('equal')
  ax.set_xlabel('easting (m)')
  ax.set_ylabel('northing (m)')
  offset = 20
  for (id, p) in sites.iteritems():
    pp.plot(p.imag, p.real, color='r', marker='o', ms=7)
    pp.text(p.imag+offset, p.real+offset, id)
  pp.plot(p_known.imag, p_known.real, color='w', marker='o', ms=7)
  pp.grid()
  pp.title("Receivers and transmitter")
  pp.savefig("%s.png" % (fn), dpi=120)
  pp.clf()
  

def mean(P, site_ids): 
  pos = np.array(P[site_ids])
  pos = pos[~np.isnan(pos)]
  return np.mean(pos)

def plot(P, fn, p_known=None):
  for site_ids in P.keys():
    if len(site_ids) > 1:
      pos = np.array(filter(lambda p: p!=None, P[site_ids]))
      #pos = pos[np.abs(pos - p_known) < 100]
      fig = pp.gcf()
      ax = fig.add_subplot(111)
      ax.axis('equal')
      ax.set_xlabel('easting (m)')
      ax.set_ylabel('northing (m)')
      pp.scatter(np.imag(pos), np.real(pos), alpha=0.3, facecolors='b', edgecolors='none', s=25)#, zorder=11)
      pp.plot(p_known.imag, p_known.real, color='w', marker='o', ms=7)
      pp.grid()
      pp.title("sites=%s total estimates=%d" % (str(site_ids), len(P)))
      pp.tight_layout()
      pp.savefig("%s-%s.png" % (fn, ''.join(map(lambda x: str(x), site_ids))), dpi=120, bbox_inches='tight')
      pp.clf()

def count(C): 
  res = {}
  for site_ids in C.keys():
    if len(site_ids) > 1:
      res[site_ids] = len(filter(lambda ell: ell!=None, C[site_ids]))
  return res

def correlation(P, C, p_known=None): 
  res = {}
  for site_ids in P.keys():
    val = []; dist = []
    if len(site_ids) > 1: 
      p_mean = mean(P, site_ids)
      for i in range(len(P[site_ids])): 
        if C[site_ids][i] is not None: 
          angle, axis0, axis1 = C[site_ids][i]
          E = position1.Ellipse(P[site_ids][i], angle, [axis0, axis1])
          val.append(E.area())
          if p_known is None: 
            dist.append(np.abs(P[site_ids][i] - p_mean))
          else:
            dist.append(np.abs(P[site_ids][i] - p_known))
      # First value is the correlation, the second is the p-value 
      # (probability of data asuuming they are uncorrelated)
      res[site_ids] = scipy.stats.stats.pearsonr(val, dist)
  return res

def coverage(P, C, p_known=None): 
  res = {}
  for site_ids in P.keys():
    val = []
    if len(site_ids) > 1: 
      p_mean = mean(P, site_ids)
      for i in range(len(P[site_ids])): 
        if C[site_ids][i] is not None: 
          angle, axis0, axis1 = C[site_ids][i]
          E = position1.Ellipse(P[site_ids][i], angle, [axis0, axis1])
          if p_known is None:
            val.append(p_mean in E)
          else: val.append(p_known in E)
      res[site_ids] = float(sum(val)) / len(val)
  return res




if __name__ == '__main__':  
  db_con = util.get_db('reader')
  
  # System params 
  sv = signal.SteeringVectors(db_con, cal_id)
  sites = util.get_sites(db_con)
  (center, zone) = util.get_center(db_con)
  
  #plot_map(site34, sites, 'beacon-map')

  #P, C = process(sv)
  #pickle.dump((P, C), open(fn+'.'+suffix, 'w'))
  (P, C) = pickle.load(open(fn+'.'+suffix, 'r'))

  plot(P, fn, site34)

  print "Count"
  for (site_ids, ct) in count(C).iteritems():
    print site_ids, '-->', ct

  print "\nCorrelation" # of distance to true position and ellipse area
  for (site_ids, r) in correlation(P, C, site34).iteritems():
    print site_ids, '--> %0.4f, p-val=%0.4f' % r

  print "\nCvg. probability" 
  for (site_ids, p) in coverage(P, C).iteritems():
    print site_ids, '--> %0.4f' % p
  

